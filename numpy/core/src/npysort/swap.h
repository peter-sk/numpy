                     // 25318 30532
#if SWAP_TYPE == 0   // 28511 30624
#define SWAP(x,y) { if (d[x] > d[y]) { type dy = d[y]; d[y] = d[x]; d[x] = dy; } }
#elif SWAP_TYPE == 1 // 21368 29538
#define SWAP(x,y) { type dx = d[x]; d[x] = dx < d[y] ? dx : d[y]; d[y] = dx < d[y] ? d[y] : dx; }
#elif SWAP_TYPE == 2 // 21255 N/A
#define SWAP(x,y) { type dx = d[x]; d[x] = dx < d[y] ? dx : d[y]; d[y] ^= dx ^ d[x]; }
#elif SWAP_TYPE == 3 // 21347 N/A
#define SWAP(x,y) { type dx = d[x]; type dy = d[y]; type tmp = d[x] = dx < dy ? dx : dy; d[y] = dy ^ dx ^ tmp; }
#elif SWAP_TYPE == 4 // 21287 29908
#define SWAP(x,y) { type dx = d[x]; type dy = d[y]; type tmp = d[x] = dx < dy ? dx : dy; cast ndy = *(cast *)&dy ^ *(cast *)&dx ^ *(cast *)&tmp; d[y] = *(type *)&ndy; }
#elif SWAP_TYPE == 5 // 21287 31087
#define SWAP(x,y) { type dx = d[x]; d[x] = dx < d[y] ? dx : d[y]; cast ndy = *(cast *)(d+y) ^ *(cast *)&dx ^ *(cast *)(d+x); d[y] = *(type *)&ndy; }
#elif SWAP_TYPE == 6 // 
#define SWAP(x,y) { type dx = d[x]; type dy = d[y]; type tmp = d[x] = dx < dy ? dx : dy; cast ndy[2]; ndy[0] = *(cast *)&dy ^ *(cast *)&dx ^ *(cast *)&tmp; ndy[1] = *(((cast *)&dy)+1) ^ *(((cast *)&dx)+1) ^ *(((cast *)&tmp)+1); d[y] = *(type *)ndy; }
#endif

#if FNS_TYPE == FNS_GUARDED
#define SWAPN(x,y) { if (y < n) { SWAP(x,y) } }
#else
#define SWAPN(x,y) SWAP(x,y)
#endif
