/* -*- c -*- */

/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario is O(N^2) so
 * the code switches to the O(NlogN) worst case heapsort if not enough progress
 * is made on the large side of the two quicksort partitions. This improves the
 * worst case while still retaining the speed of quicksort for the common case.
 * This is variant known as introsort.
 *
 *
 * def introsort(lower, higher, recursion_limit=log2(higher - lower + 1) * 2):
 *   # sort remainder with heapsort if we are not making enough progress
 *   # we arbitrarily choose 2 * log(n) as the cutoff point
 *   if recursion_limit < 0:
 *       heapsort(lower, higher)
 *       return
 *
 *   if lower < higher:
 *      pivot_pos = partition(lower, higher)
 *      # recurse into smaller first and leave larger on stack
 *      # this limits the required stack space
 *      if (pivot_pos - lower > higher - pivot_pos):
 *          quicksort(pivot_pos + 1, higher, recursion_limit - 1)
 *          quicksort(lower, pivot_pos, recursion_limit - 1)
 *      else:
 *          quicksort(lower, pivot_pos, recursion_limit - 1)
 *          quicksort(pivot_pos + 1, higher, recursion_limit - 1)
 *
 *
 * the below code implements this converted to an iteration and as an
 * additional minor optimization skips the recursion depth checking on the
 * smaller partition as it is always less than half of the remaining data and
 * will thus terminate fast enough
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_cpu_features.h"
#include "npy_sort.h"
#include "npysort_common.h"
#include "npysort_heapsort.h"
#include "numpy_tag.h"

#include "sn.h"
//#include "sn2.h"
#include "x86-qsort.h"
#include <cstdlib>
#include <utility>

#ifndef NPY_DISABLE_OPTIMIZATION
#include "x86-qsort.dispatch.h"
#endif

#define NOT_USED NPY_UNUSED(unused)
/*
 * pushing largest partition has upper bound of log2(n) space
 * we store two pointers each time
 */
#define PYA_QS_STACK (NPY_BITSOF_INTP * 2)
#define SMALL_QUICKSORT 256
#define SMALL_MERGESORT 20
#define SMALL_STRING 16

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

namespace {

template <typename Tag>
struct x86_dispatch {
    static bool quicksort(typename Tag::type *, npy_intp) { return false; }
};

template <>
struct x86_dispatch<npy::long_tag> {
    static bool quicksort(npy_int64 *start, npy_intp num)
    {
        //printf("YZA\n");
        return false;
    }
};

template <>
struct x86_dispatch<npy::int_tag> {
    static bool quicksort(npy_int *start, npy_intp num)
    {
        void (*dispfunc)(void *, npy_intp) = nullptr;
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = x86_quicksort_int);
        if (dispfunc) {
            //printf("x86_quicksort_int\n");
            (*dispfunc)(start, num);
            return true;
        }
        return false;
    }
};

template <>
struct x86_dispatch<npy::uint_tag> {
    static bool quicksort(npy_uint *start, npy_intp num)
    {
        void (*dispfunc)(void *, npy_intp) = nullptr;
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = x86_quicksort_uint);
        if (dispfunc) {
            (*dispfunc)(start, num);
            return true;
        }
        return false;
    }
};

template <>
struct x86_dispatch<npy::float_tag> {
    static bool quicksort(npy_float *start, npy_intp num)
    {
        void (*dispfunc)(void *, npy_intp) = nullptr;
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = x86_quicksort_float);
        if (dispfunc) {
            (*dispfunc)(start, num);
            return true;
        }
        return false;
    }
};

}  // namespace

template <typename Tag> static
void base_case(npy_cfloat *pl, npy_cfloat *pr) {
    npy_cfloat *pi, *pj, *pk;
    npy_cfloat vp;
    for (pi = pl + 1; pi <= pr; ++pi) {
        vp = *pi;
        pj = pi;
        pk = pi - 1;
        while (pj > pl && Tag::less(vp, *pk)) {
            *pj-- = *pk--;
        }
        *pj = vp;
    }
}

template <typename Tag, typename type>
static int
quicksort_(type *start, npy_intp num)
{
    if (x86_dispatch<Tag>::quicksort(start, num))
        return 0;
    //printf("quicksort_int_fallback\n");
    type vp;
    type *pl = start;
    type *pr = pl + num - 1;
    type *stack[PYA_QS_STACK];
    type **sptr = stack;
    type *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            heapsort_<Tag>(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (Tag::less(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            if (Tag::less(*pr, *pm)) {
                std::swap(*pr, *pm);
            }
            if (Tag::less(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            vp = *pm;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(*pi, vp));
                do {
                    --pj;
                } while (Tag::less(vp, *pj));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

static int
quicksort_int(int *start, npy_intp num)
{
    //printf("quicksort_int\n");
    int vp;
    int *pl = start;
    int *pr = pl + num - 1;
    int *stack[PYA_QS_STACK];
    int **sptr = stack;
    int *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            heapsort_<npy::int_tag>(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (*pm < *pl) {
                std::swap(*pm, *pl);
            }
            if (*pr < *pm) {
                std::swap(*pr, *pm);
            }
            if (*pm < *pl) {
                std::swap(*pm, *pl);
            }
            vp = *pm;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (*pi < vp);
                do {
                    --pj;
                } while (vp < *pj);
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }
        //sort256(pl,pr-pl+1);

        switch (pr-pl+1) {
            case 2:
            sort2<int>(pl);
            break;
            case 3:
            sort3<int>(pl);
            break;
            case 4:
            sort4<int>(pl);
            break;
            case 5:
            sort5<int>(pl);
            break;
            case 6:
            sort6<int>(pl);
            break;
            case 7:
            sort7<int>(pl);
            break;
            case 8:
            sort8<int>(pl);
            break;
            case 9:
            sort9<int>(pl);
            break;
            case 10:
            sort10<int>(pl);
            break;
            case 11:
            sort11<int>(pl);
            break;
            case 12:
            sort12<int>(pl);
            break;
            case 13:
            sort13<int>(pl);
            break;
            case 14:
            sort14<int>(pl);
            break;
            case 15:
            sort15<int>(pl);
            break;
            case 16:
            sort16<int>(pl);
            break;
            case 17:
            sort17<int>(pl);
            break;
            case 18:
            sort18<int>(pl);
            break;
            case 19:
            sort19<int>(pl);
            break;
            case 20:
            sort20<int>(pl);
            break;
            case 21:
            sort21<int>(pl);
            break;
            case 22:
            sort22<int>(pl);
            break;
            case 23:
            sort23<int>(pl);
            break;
            case 24:
            sort24<int>(pl);
            break;
            case 25:
            sort25<int>(pl);
            break;
            case 26:
            sort26<int>(pl);
            break;
            case 27:
            sort27<int>(pl);
            break;
            case 28:
            sort28<int>(pl);
            break;
            case 29:
            sort29<int>(pl);
            break;
            case 30:
            sort30<int>(pl);
            break;
            case 31:
            sort31<int>(pl);
            break;
            case 32:
            sort32<int>(pl);
            break;
            case 33:
            sort33<int>(pl);
            break;
            case 34:
            sort34<int>(pl);
            break;
            case 35:
            sort35<int>(pl);
            break;
            case 36:
            sort36<int>(pl);
            break;
            case 37:
            sort37<int>(pl);
            break;
            case 38:
            sort38<int>(pl);
            break;
            case 39:
            sort39<int>(pl);
            break;
            case 40:
            sort40<int>(pl);
            break;
            case 41:
            sort41<int>(pl);
            break;
            case 42:
            sort42<int>(pl);
            break;
            case 43:
            sort43<int>(pl);
            break;
            case 44:
            sort44<int>(pl);
            break;
            case 45:
            sort45<int>(pl);
            break;
            case 46:
            sort46<int>(pl);
            break;
            case 47:
            sort47<int>(pl);
            break;
            case 48:
            sort48<int>(pl);
            break;
            case 49:
            sort49<int>(pl);
            break;
            case 50:
            sort50<int>(pl);
            break;
            case 51:
            sort51<int>(pl);
            break;
            case 52:
            sort52<int>(pl);
            break;
            case 53:
            sort53<int>(pl);
            break;
            case 54:
            sort54<int>(pl);
            break;
            case 55:
            sort55<int>(pl);
            break;
            case 56:
            sort56<int>(pl);
            break;
            case 57:
            sort57<int>(pl);
            break;
            case 58:
            sort58<int>(pl);
            break;
            case 59:
            sort59<int>(pl);
            break;
            case 60:
            sort60<int>(pl);
            break;
            case 61:
            sort61<int>(pl);
            break;
            case 62:
            sort62<int>(pl);
            break;
            case 63:
            sort63<int>(pl);
            break;
            case 64:
            sort64<int>(pl);
            break;
/*
            case 65:
            sort65<int>(pl);
            break;
            case 66:
            sort66<int>(pl);
            break;
            case 67:
            sort67<int>(pl);
            break;
            case 68:
            sort68<int>(pl);
            break;
            case 69:
            sort69<int>(pl);
            break;
            case 70:
            sort70<int>(pl);
            break;
            case 71:
            sort71<int>(pl);
            break;
            case 72:
            sort72<int>(pl);
            break;
            case 73:
            sort73<int>(pl);
            break;
            case 74:
            sort74<int>(pl);
            break;
            case 75:
            sort75<int>(pl);
            break;
            case 76:
            sort76<int>(pl);
            break;
            case 77:
            sort77<int>(pl);
            break;
            case 78:
            sort78<int>(pl);
            break;
            case 79:
            sort79<int>(pl);
            break;
            case 80:
            sort80<int>(pl);
            break;
            case 81:
            sort81<int>(pl);
            break;
            case 82:
            sort82<int>(pl);
            break;
            case 83:
            sort83<int>(pl);
            break;
            case 84:
            sort84<int>(pl);
            break;
            case 85:
            sort85<int>(pl);
            break;
            case 86:
            sort86<int>(pl);
            break;
            case 87:
            sort87<int>(pl);
            break;
            case 88:
            sort88<int>(pl);
            break;
            case 89:
            sort89<int>(pl);
            break;
            case 90:
            sort90<int>(pl);
            break;
            case 91:
            sort91<int>(pl);
            break;
            case 92:
            sort92<int>(pl);
            break;
            case 93:
            sort93<int>(pl);
            break;
            case 94:
            sort94<int>(pl);
            break;
            case 95:
            sort95<int>(pl);
            break;
            case 96:
            sort96<int>(pl);
            break;
            case 97:
            sort97<int>(pl);
            break;
            case 98:
            sort98<int>(pl);
            break;
            case 99:
            sort99<int>(pl);
            break;
            case 100:
            sort100<int>(pl);
            break;
            case 101:
            sort101<int>(pl);
            break;
            case 102:
            sort102<int>(pl);
            break;
            case 103:
            sort103<int>(pl);
            break;
            case 104:
            sort104<int>(pl);
            break;
            case 105:
            sort105<int>(pl);
            break;
            case 106:
            sort106<int>(pl);
            break;
            case 107:
            sort107<int>(pl);
            break;
            case 108:
            sort108<int>(pl);
            break;
            case 109:
            sort109<int>(pl);
            break;
            case 110:
            sort110<int>(pl);
            break;
            case 111:
            sort111<int>(pl);
            break;
            case 112:
            sort112<int>(pl);
            break;
            case 113:
            sort113<int>(pl);
            break;
            case 114:
            sort114<int>(pl);
            break;
            case 115:
            sort115<int>(pl);
            break;
            case 116:
            sort116<int>(pl);
            break;
            case 117:
            sort117<int>(pl);
            break;
            case 118:
            sort118<int>(pl);
            break;
            case 119:
            sort119<int>(pl);
            break;
            case 120:
            sort120<int>(pl);
            break;
            case 121:
            sort121<int>(pl);
            break;
            case 122:
            sort122<int>(pl);
            break;
            case 123:
            sort123<int>(pl);
            break;
            case 124:
            sort124<int>(pl);
            break;
            case 125:
            sort125<int>(pl);
            break;
            case 126:
            sort126<int>(pl);
            break;
            case 127:
            sort127<int>(pl);
            break;
            case 128:
            sort128<int>(pl);
            break;

            case 129:
            sort129<int>(pl);
            break;
            case 130:
            sort130<int>(pl);
            break;
            case 131:
            sort131<int>(pl);
            break;
            case 132:
            sort132<int>(pl);
            break;
            case 133:
            sort133<int>(pl);
            break;
            case 134:
            sort134<int>(pl);
            break;
            case 135:
            sort135<int>(pl);
            break;
            case 136:
            sort136<int>(pl);
            break;
            case 137:
            sort137<int>(pl);
            break;
            case 138:
            sort138<int>(pl);
            break;
            case 139:
            sort139<int>(pl);
            break;
            case 140:
            sort140<int>(pl);
            break;
            case 141:
            sort141<int>(pl);
            break;
            case 142:
            sort142<int>(pl);
            break;
            case 143:
            sort143<int>(pl);
            break;
            case 144:
            sort144<int>(pl);
            break;
            case 145:
            sort145<int>(pl);
            break;
            case 146:
            sort146<int>(pl);
            break;
            case 147:
            sort147<int>(pl);
            break;
            case 148:
            sort148<int>(pl);
            break;
            case 149:
            sort149<int>(pl);
            break;
            case 150:
            sort150<int>(pl);
            break;
            case 151:
            sort151<int>(pl);
            break;
            case 152:
            sort152<int>(pl);
            break;
            case 153:
            sort153<int>(pl);
            break;
            case 154:
            sort154<int>(pl);
            break;
            case 155:
            sort155<int>(pl);
            break;
            case 156:
            sort156<int>(pl);
            break;
            case 157:
            sort157<int>(pl);
            break;
            case 158:
            sort158<int>(pl);
            break;
            case 159:
            sort159<int>(pl);
            break;
            case 160:
            sort160<int>(pl);
            break;
            case 161:
            sort161<int>(pl);
            break;
            case 162:
            sort162<int>(pl);
            break;
            case 163:
            sort163<int>(pl);
            break;
            case 164:
            sort164<int>(pl);
            break;
            case 165:
            sort165<int>(pl);
            break;
            case 166:
            sort166<int>(pl);
            break;
            case 167:
            sort167<int>(pl);
            break;
            case 168:
            sort168<int>(pl);
            break;
            case 169:
            sort169<int>(pl);
            break;
            case 170:
            sort170<int>(pl);
            break;
            case 171:
            sort171<int>(pl);
            break;
            case 172:
            sort172<int>(pl);
            break;
            case 173:
            sort173<int>(pl);
            break;
            case 174:
            sort174<int>(pl);
            break;
            case 175:
            sort175<int>(pl);
            break;
            case 176:
            sort176<int>(pl);
            break;
            case 177:
            sort177<int>(pl);
            break;
            case 178:
            sort178<int>(pl);
            break;
            case 179:
            sort179<int>(pl);
            break;
            case 180:
            sort180<int>(pl);
            break;
            case 181:
            sort181<int>(pl);
            break;
            case 182:
            sort182<int>(pl);
            break;
            case 183:
            sort183<int>(pl);
            break;
            case 184:
            sort184<int>(pl);
            break;
            case 185:
            sort185<int>(pl);
            break;
            case 186:
            sort186<int>(pl);
            break;
            case 187:
            sort187<int>(pl);
            break;
            case 188:
            sort188<int>(pl);
            break;
            case 189:
            sort189<int>(pl);
            break;
            case 190:
            sort190<int>(pl);
            break;
            case 191:
            sort191<int>(pl);
            break;
            case 192:
            sort192<int>(pl);
            break;
            case 193:
            sort193<int>(pl);
            break;
            case 194:
            sort194<int>(pl);
            break;
            case 195:
            sort195<int>(pl);
            break;
            case 196:
            sort196<int>(pl);
            break;
            case 197:
            sort197<int>(pl);
            break;
            case 198:
            sort198<int>(pl);
            break;
            case 199:
            sort199<int>(pl);
            break;
            case 200:
            sort200<int>(pl);
            break;
            case 201:
            sort201<int>(pl);
            break;
            case 202:
            sort202<int>(pl);
            break;
            case 203:
            sort203<int>(pl);
            break;
            case 204:
            sort204<int>(pl);
            break;
            case 205:
            sort205<int>(pl);
            break;
            case 206:
            sort206<int>(pl);
            break;
            case 207:
            sort207<int>(pl);
            break;
            case 208:
            sort208<int>(pl);
            break;
            case 209:
            sort209<int>(pl);
            break;
            case 210:
            sort210<int>(pl);
            break;
            case 211:
            sort211<int>(pl);
            break;
            case 212:
            sort212<int>(pl);
            break;
            case 213:
            sort213<int>(pl);
            break;
            case 214:
            sort214<int>(pl);
            break;
            case 215:
            sort215<int>(pl);
            break;
            case 216:
            sort216<int>(pl);
            break;
            case 217:
            sort217<int>(pl);
            break;
            case 218:
            sort218<int>(pl);
            break;
            case 219:
            sort219<int>(pl);
            break;
            case 220:
            sort220<int>(pl);
            break;
            case 221:
            sort221<int>(pl);
            break;
            case 222:
            sort222<int>(pl);
            break;
            case 223:
            sort223<int>(pl);
            break;
            case 224:
            sort224<int>(pl);
            break;
            case 225:
            sort225<int>(pl);
            break;
            case 226:
            sort226<int>(pl);
            break;
            case 227:
            sort227<int>(pl);
            break;
            case 228:
            sort228<int>(pl);
            break;
            case 229:
            sort229<int>(pl);
            break;
            case 230:
            sort230<int>(pl);
            break;
            case 231:
            sort231<int>(pl);
            break;
            case 232:
            sort232<int>(pl);
            break;
            case 233:
            sort233<int>(pl);
            break;
            case 234:
            sort234<int>(pl);
            break;
            case 235:
            sort235<int>(pl);
            break;
            case 236:
            sort236<int>(pl);
            break;
            case 237:
            sort237<int>(pl);
            break;
            case 238:
            sort238<int>(pl);
            break;
            case 239:
            sort239<int>(pl);
            break;
            case 240:
            sort240<int>(pl);
            break;
            case 241:
            sort241<int>(pl);
            break;
            case 242:
            sort242<int>(pl);
            break;
            case 243:
            sort243<int>(pl);
            break;
            case 244:
            sort244<int>(pl);
            break;
            case 245:
            sort245<int>(pl);
            break;
            case 246:
            sort246<int>(pl);
            break;
            case 247:
            sort247<int>(pl);
            break;
            case 248:
            sort248<int>(pl);
            break;
            case 249:
            sort249<int>(pl);
            break;
            case 250:
            sort250<int>(pl);
            break;
            case 251:
            sort251<int>(pl);
            break;
            case 252:
            sort252<int>(pl);
            break;
            case 253:
            sort253<int>(pl);
            break;
            case 254:
            sort254<int>(pl);
            break;
            case 255:
            sort255<int>(pl);
            break;
            case 256:
            sort256<int>(pl);
            break;
*/
                default:
                  break;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

template <typename Tag, typename type>
static int
aquicksort_(type *vv, npy_intp *tosort, npy_intp num)
{
    type *v = vv;
    type vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            aheapsort_<Tag>(vv, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (Tag::less(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            if (Tag::less(v[*pr], v[*pm])) {
                std::swap(*pr, *pm);
            }
            if (Tag::less(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(v[*pi], vp));
                do {
                    --pj;
                } while (Tag::less(vp, v[*pj]));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

/*
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
static int
string_quicksort_(type *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    const size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    type *vp;
    type *pl = start;
    type *pr = pl + (num - 1) * len;
    type *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    vp = (type *)malloc(PyArray_ITEMSIZE(arr));
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            string_heapsort_<Tag>(pl, (pr - pl) / len + 1, varr);
            goto stack_pop;
        }
        while ((size_t)(pr - pl) > SMALL_QUICKSORT * len) {
            /* quicksort partition */
            pm = pl + (((pr - pl) / len) >> 1) * len;
            if (Tag::less(pm, pl, len)) {
                Tag::swap(pm, pl, len);
            }
            if (Tag::less(pr, pm, len)) {
                Tag::swap(pr, pm, len);
            }
            if (Tag::less(pm, pl, len)) {
                Tag::swap(pm, pl, len);
            }
            Tag::copy(vp, pm, len);
            pi = pl;
            pj = pr - len;
            Tag::swap(pm, pj, len);
            for (;;) {
                do {
                    pi += len;
                } while (Tag::less(pi, vp, len));
                do {
                    pj -= len;
                } while (Tag::less(vp, pj, len));
                if (pi >= pj) {
                    break;
                }
                Tag::swap(pi, pj, len);
            }
            pk = pr - len;
            Tag::swap(pi, pk, len);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + len;
                *sptr++ = pr;
                pr = pi - len;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - len;
                pl = pi + len;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + len; pi <= pr; pi += len) {
            Tag::copy(vp, pi, len);
            pj = pi;
            pk = pi - len;
            while (pj > pl && Tag::less(vp, pk, len)) {
                Tag::copy(pj, pk, len);
                pj -= len;
                pk -= len;
            }
            Tag::copy(pj, vp, len);
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    free(vp);
    return 0;
}

template <typename Tag, typename type>
static int
string_aquicksort_(type *vv, npy_intp *tosort, npy_intp num, void *varr)
{
    type *v = vv;
    PyArrayObject *arr = (PyArrayObject *)varr;
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    type *vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            string_aheapsort_<Tag>(vv, pl, pr - pl + 1, varr);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (Tag::less(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            if (Tag::less(v + (*pr) * len, v + (*pm) * len, len)) {
                std::swap(*pr, *pm);
            }
            if (Tag::less(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            vp = v + (*pm) * len;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(v + (*pi) * len, vp, len));
                do {
                    --pj;
                } while (Tag::less(vp, v + (*pj) * len, len));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v + vi * len;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, v + (*pk) * len, len)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

NPY_NO_EXPORT int
npy_quicksort(void *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
    char *vp;
    char *pl = (char *)start;
    char *pr = pl + (num - 1) * elsize;
    char *stack[PYA_QS_STACK];
    char **sptr = stack;
    char *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    vp = (char *)malloc(elsize);
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_heapsort(pl, (pr - pl) / elsize + 1, varr);
            goto stack_pop;
        }
        while (pr - pl > SMALL_QUICKSORT * elsize) {
            /* quicksort partition */
            pm = pl + (((pr - pl) / elsize) >> 1) * elsize;
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            if (cmp(pr, pm, arr) < 0) {
                GENERIC_SWAP(pr, pm, elsize);
            }
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            GENERIC_COPY(vp, pm, elsize);
            pi = pl;
            pj = pr - elsize;
            GENERIC_SWAP(pm, pj, elsize);
            /*
             * Generic comparisons may be buggy, so don't rely on the sentinels
             * to keep the pointers from going out of bounds.
             */
            for (;;) {
                do {
                    pi += elsize;
                } while (cmp(pi, vp, arr) < 0 && pi < pj);
                do {
                    pj -= elsize;
                } while (cmp(vp, pj, arr) < 0 && pi < pj);
                if (pi >= pj) {
                    break;
                }
                GENERIC_SWAP(pi, pj, elsize);
            }
            pk = pr - elsize;
            GENERIC_SWAP(pi, pk, elsize);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + elsize;
                *sptr++ = pr;
                pr = pi - elsize;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - elsize;
                pl = pi + elsize;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + elsize; pi <= pr; pi += elsize) {
            GENERIC_COPY(vp, pi, elsize);
            pj = pi;
            pk = pi - elsize;
            while (pj > pl && cmp(vp, pk, arr) < 0) {
                GENERIC_COPY(pj, pk, elsize);
                pj -= elsize;
                pk -= elsize;
            }
            GENERIC_COPY(pj, vp, elsize);
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    free(vp);
    return 0;
}

NPY_NO_EXPORT int
npy_aquicksort(void *vv, npy_intp *tosort, npy_intp num, void *varr)
{
    char *v = (char *)vv;
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
    char *vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_aheapsort(vv, pl, pr - pl + 1, varr);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            if (cmp(v + (*pr) * elsize, v + (*pm) * elsize, arr) < 0) {
                INTP_SWAP(*pr, *pm);
            }
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            vp = v + (*pm) * elsize;
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (cmp(v + (*pi) * elsize, vp, arr) < 0 && pi < pj);
                do {
                    --pj;
                } while (cmp(vp, v + (*pj) * elsize, arr) < 0 && pi < pj);
                if (pi >= pj) {
                    break;
                }
                INTP_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v + vi * elsize;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && cmp(vp, v + (*pk) * elsize, arr) < 0) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

/***************************************
 * C > C++ dispatch
 ***************************************/

NPY_NO_EXPORT int
quicksort_bool(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::bool_tag>((npy_bool *)start, n);
}
NPY_NO_EXPORT int
quicksort_byte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::byte_tag>((npy_byte *)start, n);
}
NPY_NO_EXPORT int
quicksort_ubyte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::ubyte_tag>((npy_ubyte *)start, n);
}
NPY_NO_EXPORT int
quicksort_short(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::short_tag>((npy_short *)start, n);
}
NPY_NO_EXPORT int
quicksort_ushort(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::ushort_tag>((npy_ushort *)start, n);
}
NPY_NO_EXPORT int
quicksort_int(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    //return quicksort_int((npy_int *)start, n);
    return quicksort_<npy::int_tag>((npy_int *)start, n);
}
NPY_NO_EXPORT int
quicksort_uint(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::uint_tag>((npy_uint *)start, n);
}
NPY_NO_EXPORT int
quicksort_long(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::long_tag>((npy_long *)start, n);
}
NPY_NO_EXPORT int
quicksort_ulong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::ulong_tag>((npy_ulong *)start, n);
}
NPY_NO_EXPORT int
quicksort_longlong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::longlong_tag>((npy_longlong *)start, n);
}
NPY_NO_EXPORT int
quicksort_ulonglong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::ulonglong_tag>((npy_ulonglong *)start, n);
}
NPY_NO_EXPORT int
quicksort_half(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::half_tag>((npy_half *)start, n);
}
NPY_NO_EXPORT int
quicksort_float(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::float_tag>((npy_float *)start, n);
}
NPY_NO_EXPORT int
quicksort_double(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::double_tag>((npy_double *)start, n);
}
NPY_NO_EXPORT int
quicksort_longdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::longdouble_tag>((npy_longdouble *)start, n);
}
NPY_NO_EXPORT int
quicksort_cfloat(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::cfloat_tag>((npy_cfloat *)start, n);
}
NPY_NO_EXPORT int
quicksort_cdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::cdouble_tag>((npy_cdouble *)start, n);
}
NPY_NO_EXPORT int
quicksort_clongdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::clongdouble_tag>((npy_clongdouble *)start, n);
}
NPY_NO_EXPORT int
quicksort_datetime(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::datetime_tag>((npy_datetime *)start, n);
}
NPY_NO_EXPORT int
quicksort_timedelta(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::timedelta_tag>((npy_timedelta *)start, n);
}

NPY_NO_EXPORT int
aquicksort_bool(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::bool_tag>((npy_bool *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_byte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::byte_tag>((npy_byte *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ubyte(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ubyte_tag>((npy_ubyte *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_short(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::short_tag>((npy_short *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ushort(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ushort_tag>((npy_ushort *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_int(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::int_tag>((npy_int *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_uint(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::uint_tag>((npy_uint *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_long(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::long_tag>((npy_long *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ulong(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ulong_tag>((npy_ulong *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_longlong(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::longlong_tag>((npy_longlong *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ulonglong(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ulonglong_tag>((npy_ulonglong *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_half(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::half_tag>((npy_half *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_float(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::float_tag>((npy_float *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_double(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::double_tag>((npy_double *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_longdouble(void *vv, npy_intp *tosort, npy_intp n,
                      void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::longdouble_tag>((npy_longdouble *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_cfloat(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::cfloat_tag>((npy_cfloat *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_cdouble(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::cdouble_tag>((npy_cdouble *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_clongdouble(void *vv, npy_intp *tosort, npy_intp n,
                       void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::clongdouble_tag>((npy_clongdouble *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_datetime(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::datetime_tag>((npy_datetime *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_timedelta(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::timedelta_tag>((npy_timedelta *)vv, tosort, n);
}

NPY_NO_EXPORT int
quicksort_string(void *start, npy_intp n, void *varr)
{
    return string_quicksort_<npy::string_tag>((npy_char *)start, n, varr);
}
NPY_NO_EXPORT int
quicksort_unicode(void *start, npy_intp n, void *varr)
{
    return string_quicksort_<npy::unicode_tag>((npy_ucs4 *)start, n, varr);
}

NPY_NO_EXPORT int
aquicksort_string(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return string_aquicksort_<npy::string_tag>((npy_char *)vv, tosort, n,
                                               varr);
}
NPY_NO_EXPORT int
aquicksort_unicode(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return string_aquicksort_<npy::unicode_tag>((npy_ucs4 *)vv, tosort, n,
                                                varr);
}
