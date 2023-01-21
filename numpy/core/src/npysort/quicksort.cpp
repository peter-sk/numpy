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

#include "x86-qsort.h"
#include <cstdlib>
#include <cstring>
#include <utility>

#ifndef NPY_DISABLE_OPTIMIZATION
#include "x86-qsort.dispatch.h"
#endif

#include "npysort_config.h"
#if NPY_SORT_TYPE == NPY_SORT_FNS
#if FNS_TYPE == FNS_ALL
#include "sn.h"
#else
#include "sn2.h"
#endif
#endif
#ifdef NPY_SORT_DEBUG
static bool npy_sort_debug = true;
#endif

#define NOT_USED NPY_UNUSED(unused)
/*
 * pushing largest partition has upper bound of log2(n) space
 * we store two pointers each time
 */
#define PYA_QS_STACK (NPY_BITSOF_INTP * 2)
#define SMALL_QUICKSORT NPY_SORT_BASE
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
#ifdef NPY_SORT_DEBUG
            if (npy_sort_debug) {
                npy_sort_debug = false;
                printf("%d avx512_bitonic",NPY_SORT_BASE);
                fflush(stdout);
            }
#endif
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
#ifdef NPY_SORT_DEBUG
            if (npy_sort_debug) {
                npy_sort_debug = false;
                printf("%d avx512_bitonic",NPY_SORT_BASE);
                fflush(stdout);
            }
#endif
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
#ifdef NPY_SORT_DEBUG
            if (npy_sort_debug) {
                npy_sort_debug = false;
                printf("%d avx512_bitonic",NPY_SORT_BASE);
                fflush(stdout);
            }
#endif
            (*dispfunc)(start, num);
            return true;
        }
        return false;
    }
};

}  // namespace

template <typename Tag, typename type>
static int
quicksort_(type *start, npy_intp num)
{
#if NPY_SORT_TYPE == NPY_SORT_DISPATCH
    if (x86_dispatch<Tag>::quicksort(start, num))
        return 0;
#endif
#ifdef NPY_SORT_DEBUG
    if (npy_sort_debug) {
        npy_sort_debug = false;
        printf("%d fallback ",NPY_SORT_BASE);
        fflush(stdout);
    }
#endif
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
        while ((pr - pl + 1) > SMALL_QUICKSORT) {
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

#ifdef NPY_SORT_CHECK
    for (int i = 1; i < num; i++) {
        if (Tag::less(start[i],start[i-1])) {
            printf("value at %d larger than value at %d\n",i-1,i);
        }
    }
#endif
    return 0;
}

#if NPY_SORT_TYPE == NPY_SORT_FNS
template <typename type, typename cast> static int
quicksort_fns(type *start, npy_intp num)
{
#ifdef NPY_SORT_DEBUG
    if (npy_sort_debug) {
        npy_sort_debug = false;
        printf("%d fns_%s ",NPY_SORT_BASE,
#if FNS_TYPE == FNS_ALL
        "all"
#elif FNS_TYPE == FNS_GUARDED
        "guarded"
#elif FNS_TYPE == FNS_COPY
        "copy"
#elif FNS_TYPE == FNS_OVERLAP
        "overlap"
#endif
        );
        fflush(stdout);
    }
#endif
#if FNS_TYPE == FNS_COPY
    type max_int = std::numeric_limits<type>::max();
    type temp[NPY_SORT_POWER+1];
    int n, i;
#endif
    type vp;
    type *pl = start;
    type *pr = pl + num - 1;
    type *stack[PYA_QS_STACK];
    type **sptr = stack;
    type *pm, *pi, *pj, *pk;
    type depth[PYA_QS_STACK];
    type *psdepth = depth;
    type cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            heapsort_<npy::int_tag>(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl + 1) > SMALL_QUICKSORT) {
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
#if FNS_TYPE == FNS_COPY
        n = pr-pl+1;
        std::memcpy(temp,pl,n*sizeof(type));
        for (i = n; i < NPY_SORT_POWER; i++) {
            temp[i] = max_int;
        }
        NPY_SORT_FUNC<type, cast>(temp,n);
        std::memcpy(pl,temp,n*sizeof(type));
#elif FNS_TYPE == FNS_GUARDED
#ifdef NPY_SORT_DEBUG
        //printf("sort between %ld and %ld with length %ld and SN of size %d overlapping until %ld\n",pl-start,pr-start,pr-pl+1,NPY_SORT_POWER,pl-start+NPY_SORT_POWER);
#endif
        NPY_SORT_FUNC<type, cast>(pl,pr-pl+1);
#elif FNS_TYPE == FNS_OVERLAP
        if (pl-start+NPY_SORT_POWER < num) {
            NPY_SORT_FUNC<type, cast>(pl,pr-pl+1);
        } else {
            /* insertion sort */
            for (pi = pl + 1; pi <= pr; ++pi) {
                vp = *pi;
                pj = pi;
                pk = pi - 1;
                while (pj > pl && vp < *pk) {
                    *pj-- = *pk--;
                }
                *pj = vp;
            }
        }
#elif FNS_TYPE == FNS_ALL
        switch (pr-pl+1) {
#if NPY_SORT_BASE > 1
            case 2:
            sort_2<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 2
            case 3:
            sort_3<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 3
            case 4:
            sort_4<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 4
            case 5:
            sort_5<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 5
            case 6:
            sort_6<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 6
            case 7:
            sort_7<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 7
            case 8:
            sort_8<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 8
            case 9:
            sort_9<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 9
            case 10:
            sort_10<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 10
            case 11:
            sort_11<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 11
            case 12:
            sort_12<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 12
            case 13:
            sort_13<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 13
            case 14:
            sort_14<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 14
            case 15:
            sort_15<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 15
            case 16:
            sort_16<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 16
            case 17:
            sort_17<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 17
            case 18:
            sort_18<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 18
            case 19:
            sort_19<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 19
            case 20:
            sort_20<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 20
            case 21:
            sort_21<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 21
            case 22:
            sort_22<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 22
            case 23:
            sort_23<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 23
            case 24:
            sort_24<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 24
            case 25:
            sort_25<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 25
            case 26:
            sort_26<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 26
            case 27:
            sort_27<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 27
            case 28:
            sort_28<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 28
            case 29:
            sort_29<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 29
            case 30:
            sort_30<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 30
            case 31:
            sort_31<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 31
            case 32:
            sort_32<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 32
            case 33:
            sort_33<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 33
            case 34:
            sort_34<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 34
            case 35:
            sort_35<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 35
            case 36:
            sort_36<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 36
            case 37:
            sort_37<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 37
            case 38:
            sort_38<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 38
            case 39:
            sort_39<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 39
            case 40:
            sort_40<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 40
            case 41:
            sort_41<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 41
            case 42:
            sort_42<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 42
            case 43:
            sort_43<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 43
            case 44:
            sort_44<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 44
            case 45:
            sort_45<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 45
            case 46:
            sort_46<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 46
            case 47:
            sort_47<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 47
            case 48:
            sort_48<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 48
            case 49:
            sort_49<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 49
            case 50:
            sort_50<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 50
            case 51:
            sort_51<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 51
            case 52:
            sort_52<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 52
            case 53:
            sort_53<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 53
            case 54:
            sort_54<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 54
            case 55:
            sort_55<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 55
            case 56:
            sort_56<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 56
            case 57:
            sort_57<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 57
            case 58:
            sort_58<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 58
            case 59:
            sort_59<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 59
            case 60:
            sort_60<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 60
            case 61:
            sort_61<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 61
            case 62:
            sort_62<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 62
            case 63:
            sort_63<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 63
            case 64:
            sort_64<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 64
            case 65:
            sort_65<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 65
            case 66:
            sort_66<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 66
            case 67:
            sort_67<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 67
            case 68:
            sort_68<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 68
            case 69:
            sort_69<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 69
            case 70:
            sort_70<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 70
            case 71:
            sort_71<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 71
            case 72:
            sort_72<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 72
            case 73:
            sort_73<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 73
            case 74:
            sort_74<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 74
            case 75:
            sort_75<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 75
            case 76:
            sort_76<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 76
            case 77:
            sort_77<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 77
            case 78:
            sort_78<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 78
            case 79:
            sort_79<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 79
            case 80:
            sort_80<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 80
            case 81:
            sort_81<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 81
            case 82:
            sort_82<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 82
            case 83:
            sort_83<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 83
            case 84:
            sort_84<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 84
            case 85:
            sort_85<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 85
            case 86:
            sort_86<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 86
            case 87:
            sort_87<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 87
            case 88:
            sort_88<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 88
            case 89:
            sort_89<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 89
            case 90:
            sort_90<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 90
            case 91:
            sort_91<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 91
            case 92:
            sort_92<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 92
            case 93:
            sort_93<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 93
            case 94:
            sort_94<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 94
            case 95:
            sort_95<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 95
            case 96:
            sort_96<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 96
            case 97:
            sort_97<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 97
            case 98:
            sort_98<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 98
            case 99:
            sort_99<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 99
            case 100:
            sort_100<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 100
            case 101:
            sort_101<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 101
            case 102:
            sort_102<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 102
            case 103:
            sort_103<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 103
            case 104:
            sort_104<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 104
            case 105:
            sort_105<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 105
            case 106:
            sort_106<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 106
            case 107:
            sort_107<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 107
            case 108:
            sort_108<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 108
            case 109:
            sort_109<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 109
            case 110:
            sort_110<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 110
            case 111:
            sort_111<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 111
            case 112:
            sort_112<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 112
            case 113:
            sort_113<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 113
            case 114:
            sort_114<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 114
            case 115:
            sort_115<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 115
            case 116:
            sort_116<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 116
            case 117:
            sort_117<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 117
            case 118:
            sort_118<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 118
            case 119:
            sort_119<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 119
            case 120:
            sort_120<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 120
            case 121:
            sort_121<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 121
            case 122:
            sort_122<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 122
            case 123:
            sort_123<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 123
            case 124:
            sort_124<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 124
            case 125:
            sort_125<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 125
            case 126:
            sort_126<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 126
            case 127:
            sort_127<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 127
            case 128:
            sort_128<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 128
            case 129:
            sort_129<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 129
            case 130:
            sort_130<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 130
            case 131:
            sort_131<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 131
            case 132:
            sort_132<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 132
            case 133:
            sort_133<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 133
            case 134:
            sort_134<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 134
            case 135:
            sort_135<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 135
            case 136:
            sort_136<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 136
            case 137:
            sort_137<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 137
            case 138:
            sort_138<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 138
            case 139:
            sort_139<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 139
            case 140:
            sort_140<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 140
            case 141:
            sort_141<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 141
            case 142:
            sort_142<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 142
            case 143:
            sort_143<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 143
            case 144:
            sort_144<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 144
            case 145:
            sort_145<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 145
            case 146:
            sort_146<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 146
            case 147:
            sort_147<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 147
            case 148:
            sort_148<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 148
            case 149:
            sort_149<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 149
            case 150:
            sort_150<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 150
            case 151:
            sort_151<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 151
            case 152:
            sort_152<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 152
            case 153:
            sort_153<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 153
            case 154:
            sort_154<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 154
            case 155:
            sort_155<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 155
            case 156:
            sort_156<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 156
            case 157:
            sort_157<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 157
            case 158:
            sort_158<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 158
            case 159:
            sort_159<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 159
            case 160:
            sort_160<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 160
            case 161:
            sort_161<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 161
            case 162:
            sort_162<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 162
            case 163:
            sort_163<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 163
            case 164:
            sort_164<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 164
            case 165:
            sort_165<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 165
            case 166:
            sort_166<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 166
            case 167:
            sort_167<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 167
            case 168:
            sort_168<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 168
            case 169:
            sort_169<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 169
            case 170:
            sort_170<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 170
            case 171:
            sort_171<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 171
            case 172:
            sort_172<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 172
            case 173:
            sort_173<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 173
            case 174:
            sort_174<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 174
            case 175:
            sort_175<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 175
            case 176:
            sort_176<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 176
            case 177:
            sort_177<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 177
            case 178:
            sort_178<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 178
            case 179:
            sort_179<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 179
            case 180:
            sort_180<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 180
            case 181:
            sort_181<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 181
            case 182:
            sort_182<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 182
            case 183:
            sort_183<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 183
            case 184:
            sort_184<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 184
            case 185:
            sort_185<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 185
            case 186:
            sort_186<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 186
            case 187:
            sort_187<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 187
            case 188:
            sort_188<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 188
            case 189:
            sort_189<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 189
            case 190:
            sort_190<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 190
            case 191:
            sort_191<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 191
            case 192:
            sort_192<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 192
            case 193:
            sort_193<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 193
            case 194:
            sort_194<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 194
            case 195:
            sort_195<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 195
            case 196:
            sort_196<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 196
            case 197:
            sort_197<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 197
            case 198:
            sort_198<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 198
            case 199:
            sort_199<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 199
            case 200:
            sort_200<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 200
            case 201:
            sort_201<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 201
            case 202:
            sort_202<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 202
            case 203:
            sort_203<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 203
            case 204:
            sort_204<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 204
            case 205:
            sort_205<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 205
            case 206:
            sort_206<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 206
            case 207:
            sort_207<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 207
            case 208:
            sort_208<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 208
            case 209:
            sort_209<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 209
            case 210:
            sort_210<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 210
            case 211:
            sort_211<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 211
            case 212:
            sort_212<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 212
            case 213:
            sort_213<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 213
            case 214:
            sort_214<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 214
            case 215:
            sort_215<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 215
            case 216:
            sort_216<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 216
            case 217:
            sort_217<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 217
            case 218:
            sort_218<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 218
            case 219:
            sort_219<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 219
            case 220:
            sort_220<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 220
            case 221:
            sort_221<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 221
            case 222:
            sort_222<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 222
            case 223:
            sort_223<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 223
            case 224:
            sort_224<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 224
            case 225:
            sort_225<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 225
            case 226:
            sort_226<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 226
            case 227:
            sort_227<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 227
            case 228:
            sort_228<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 228
            case 229:
            sort_229<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 229
            case 230:
            sort_230<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 230
            case 231:
            sort_231<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 231
            case 232:
            sort_232<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 232
            case 233:
            sort_233<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 233
            case 234:
            sort_234<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 234
            case 235:
            sort_235<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 235
            case 236:
            sort_236<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 236
            case 237:
            sort_237<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 237
            case 238:
            sort_238<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 238
            case 239:
            sort_239<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 239
            case 240:
            sort_240<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 240
            case 241:
            sort_241<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 241
            case 242:
            sort_242<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 242
            case 243:
            sort_243<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 243
            case 244:
            sort_244<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 244
            case 245:
            sort_245<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 245
            case 246:
            sort_246<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 246
            case 247:
            sort_247<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 247
            case 248:
            sort_248<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 248
            case 249:
            sort_249<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 249
            case 250:
            sort_250<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 250
            case 251:
            sort_251<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 251
            case 252:
            sort_252<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 252
            case 253:
            sort_253<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 253
            case 254:
            sort_254<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 254
            case 255:
            sort_255<type,cast>(pl);
            break;
#endif
#if NPY_SORT_BASE > 255
            case 256:
            sort_256<type,cast>(pl);
            break;
                default:
                  break;
#endif
        }
#endif
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

#ifdef NPY_SORT_CHECK
    for (int i = 1; i < num; i++) {
        if (start[i] < start[i-1]) {
            printf("value at %d larger than value at %d\n",i-1,i);
        }
    }
#endif
    return 0;
}
#endif

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
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_bool,npy_bool>((npy_bool *)start, n);
#else
    return quicksort_<npy::bool_tag>((npy_bool *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_byte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_byte,npy_byte>((npy_byte *)start, n);
#else
    return quicksort_<npy::byte_tag>((npy_byte *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_ubyte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_ubyte,npy_ubyte>((npy_ubyte *)start, n);
#else
    return quicksort_<npy::ubyte_tag>((npy_ubyte *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_short(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_short,npy_short>((npy_short *)start, n);
#else
    return quicksort_<npy::short_tag>((npy_short *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_ushort(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_ushort,npy_ushort>((npy_ushort *)start, n);
#else
    return quicksort_<npy::ushort_tag>((npy_ushort *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_int(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_int,npy_int>((npy_int *)start, n);
#else
    return quicksort_<npy::int_tag>((npy_int *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_uint(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_uint,npy_uint>((npy_uint *)start, n);
#else
    return quicksort_<npy::uint_tag>((npy_uint *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_long(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_long,npy_long>((npy_long *)start, n);
#else
    return quicksort_<npy::long_tag>((npy_long *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_ulong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_ulong,npy_ulong>((npy_ulong *)start, n);
#else
    return quicksort_<npy::ulong_tag>((npy_ulong *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_longlong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_longlong,npy_longlong>((npy_longlong *)start, n);
#else
    return quicksort_<npy::longlong_tag>((npy_longlong *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_ulonglong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_ulonglong,npy_ulonglong>((npy_ulonglong *)start, n);
#else
    return quicksort_<npy::ulonglong_tag>((npy_ulonglong *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_half(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_half,npy_ushort>((npy_half *)start, n);
#else
    return quicksort_<npy::half_tag>((npy_half *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_float(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_float,npy_uint>((npy_float *)start, n);
#else
    return quicksort_<npy::float_tag>((npy_float *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_double(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_double,npy_ulong>((npy_double *)start, n);
#else
    return quicksort_<npy::double_tag>((npy_double *)start, n);
#endif
}
NPY_NO_EXPORT int
quicksort_longdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
#if NPY_SORT_TYPE == NPY_SORT_FNS
    return quicksort_fns<npy_longdouble,npy_ulonglong>((npy_longdouble *)start, n);
#else
    return quicksort_<npy::longdouble_tag>((npy_longdouble *)start, n);
#endif
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

