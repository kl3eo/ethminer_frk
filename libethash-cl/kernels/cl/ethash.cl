// Copyright 2017 Yurio Miyazawa (a.k.a zawawa) <me@yurio.net>
//
// This file is part of Gateless Gate Sharp.
//
// Gateless Gate Sharp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Gateless Gate Sharp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Gateless Gate Sharp.  If not, see <http://www.gnu.org/licenses/>.

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_AMD     1
#define OPENCL_PLATFORM_CLOVER  2
#define OPENCL_PLATFORM_NVIDIA  3
#define OPENCL_PLATFORM_INTEL   4

#if (defined(__Tahiti__) || defined(__Pitcairn__) || defined(__Capeverde__) || defined(__Oland__) || defined(__Hainan__))
#define LEGACY
#endif

#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif

#if defined(cl_amd_media_ops)
#if PLATFORM == OPENCL_PLATFORM_CLOVER
/*
 * MESA define cl_amd_media_ops but no amd_bitalign() defined.
 * https://github.com/openwall/john/issues/3454#issuecomment-436899959
 */
uint2 amd_bitalign(uint2 src0, uint2 src1, uint2 src2)
{
    uint2 dst;
    __asm("v_alignbit_b32 %0, %2, %3, %4\n"
          "v_alignbit_b32 %1, %5, %6, %7"
          : "=v" (dst.x), "=v" (dst.y)
          : "v" (src0.x), "v" (src1.x), "v" (src2.x),
            "v" (src0.y), "v" (src1.y), "v" (src2.y));
    return dst;
}
#endif
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#elif defined(cl_nv_pragma_unroll)
uint amd_bitalign(uint src0, uint src1, uint src2)
{
    uint dest;
    asm("shf.r.wrap.b32 %0, %2, %1, %3;" : "=r"(dest) : "r"(src0), "r"(src1), "r"(src2));
    return dest;
}
#else
#define amd_bitalign(src0, src1, src2) ((uint) (((((ulong)(src0)) << 32) | (ulong)(src1)) >> ((src2) & 31)))
#endif

#if WORKSIZE % 4 != 0
#error "WORKSIZE has to be a multiple of 4"
#endif

#define FNV_PRIME 0x01000193U

static __constant uint2 const Keccak_f1600_RC[24] = {
    (uint2)(0x00000001, 0x00000000),
    (uint2)(0x00008082, 0x00000000),
    (uint2)(0x0000808a, 0x80000000),
    (uint2)(0x80008000, 0x80000000),
    (uint2)(0x0000808b, 0x00000000),
    (uint2)(0x80000001, 0x00000000),
    (uint2)(0x80008081, 0x80000000),
    (uint2)(0x00008009, 0x80000000),
    (uint2)(0x0000008a, 0x00000000),
    (uint2)(0x00000088, 0x00000000),
    (uint2)(0x80008009, 0x00000000),
    (uint2)(0x8000000a, 0x00000000),
    (uint2)(0x8000808b, 0x00000000),
    (uint2)(0x0000008b, 0x80000000),
    (uint2)(0x00008089, 0x80000000),
    (uint2)(0x00008003, 0x80000000),
    (uint2)(0x00008002, 0x80000000),
    (uint2)(0x00000080, 0x80000000),
    (uint2)(0x0000800a, 0x00000000),
    (uint2)(0x8000000a, 0x80000000),
    (uint2)(0x80008081, 0x80000000),
    (uint2)(0x00008080, 0x80000000),
    (uint2)(0x80000001, 0x00000000),
    (uint2)(0x80008008, 0x80000000),
};

#ifdef cl_amd_media_ops

#ifdef LEGACY
#define barrier(x) mem_fence(x)
#endif

#define ROTL64_1(x, y) amd_bitalign((x), (x).s10, 32 - (y))
#define ROTL64_2(x, y) amd_bitalign((x).s10, (x), 32 - (y))

#else

#define ROTL64_1(x, y) as_uint2(rotate(as_ulong(x), (ulong)(y)))
#define ROTL64_2(x, y) ROTL64_1(x, (y) + 32)

#endif


#define KECCAKF_1600_RND(a, i, outsz) do { \
    const uint2 m0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20] ^ ROTL64_1(a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22], 1);\
    const uint2 m1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21] ^ ROTL64_1(a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23], 1);\
    const uint2 m2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22] ^ ROTL64_1(a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24], 1);\
    const uint2 m3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23] ^ ROTL64_1(a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20], 1);\
    const uint2 m4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24] ^ ROTL64_1(a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21], 1);\
    \
    const uint2 tmp = a[1]^m0;\
    \
    a[0] ^= m4;\
    a[5] ^= m4; \
    a[10] ^= m4; \
    a[15] ^= m4; \
    a[20] ^= m4; \
    \
    a[6] ^= m0; \
    a[11] ^= m0; \
    a[16] ^= m0; \
    a[21] ^= m0; \
    \
    a[2] ^= m1; \
    a[7] ^= m1; \
    a[12] ^= m1; \
    a[17] ^= m1; \
    a[22] ^= m1; \
    \
    a[3] ^= m2; \
    a[8] ^= m2; \
    a[13] ^= m2; \
    a[18] ^= m2; \
    a[23] ^= m2; \
    \
    a[4] ^= m3; \
    a[9] ^= m3; \
    a[14] ^= m3; \
    a[19] ^= m3; \
    a[24] ^= m3; \
    \
    a[1] = ROTL64_2(a[6], 12);\
    a[6] = ROTL64_1(a[9], 20);\
    a[9] = ROTL64_2(a[22], 29);\
    a[22] = ROTL64_2(a[14], 7);\
    a[14] = ROTL64_1(a[20], 18);\
    a[20] = ROTL64_2(a[2], 30);\
    a[2] = ROTL64_2(a[12], 11);\
    a[12] = ROTL64_1(a[13], 25);\
    a[13] = ROTL64_1(a[19],  8);\
    a[19] = ROTL64_2(a[23], 24);\
    a[23] = ROTL64_2(a[15], 9);\
    a[15] = ROTL64_1(a[4], 27);\
    a[4] = ROTL64_1(a[24], 14);\
    a[24] = ROTL64_1(a[21],  2);\
    a[21] = ROTL64_2(a[8], 23);\
    a[8] = ROTL64_2(a[16], 13);\
    a[16] = ROTL64_2(a[5], 4);\
    a[5] = ROTL64_1(a[3], 28);\
    a[3] = ROTL64_1(a[18], 21);\
    a[18] = ROTL64_1(a[17], 15);\
    a[17] = ROTL64_1(a[11], 10);\
    a[11] = ROTL64_1(a[7],  6);\
    a[7] = ROTL64_1(a[10],  3);\
    a[10] = ROTL64_1(tmp,  1);\
    \
    uint2 m5 = a[0]; uint2 m6 = a[1]; a[0] = bitselect(a[0]^a[2],a[0],a[1]); \
    a[0] ^= as_uint2(Keccak_f1600_RC[i]); \
    if (outsz > 1) { \
        a[1] = bitselect(a[1]^a[3],a[1],a[2]); a[2] = bitselect(a[2]^a[4],a[2],a[3]); a[3] = bitselect(a[3]^m5,a[3],a[4]); a[4] = bitselect(a[4]^m6,a[4],m5);\
        if (outsz > 4) { \
            m5 = a[5]; m6 = a[6]; a[5] = bitselect(a[5]^a[7],a[5],a[6]); a[6] = bitselect(a[6]^a[8],a[6],a[7]); a[7] = bitselect(a[7]^a[9],a[7],a[8]); a[8] = bitselect(a[8]^m5,a[8],a[9]); a[9] = bitselect(a[9]^m6,a[9],m5);\
            if (outsz > 8) { \
                m5 = a[10]; m6 = a[11]; a[10] = bitselect(a[10]^a[12],a[10],a[11]); a[11] = bitselect(a[11]^a[13],a[11],a[12]); a[12] = bitselect(a[12]^a[14],a[12],a[13]); a[13] = bitselect(a[13]^m5,a[13],a[14]); a[14] = bitselect(a[14]^m6,a[14],m5);\
                m5 = a[15]; m6 = a[16]; a[15] = bitselect(a[15]^a[17],a[15],a[16]); a[16] = bitselect(a[16]^a[18],a[16],a[17]); a[17] = bitselect(a[17]^a[19],a[17],a[18]); a[18] = bitselect(a[18]^m5,a[18],a[19]); a[19] = bitselect(a[19]^m6,a[19],m5);\
                m5 = a[20]; m6 = a[21]; a[20] = bitselect(a[20]^a[22],a[20],a[21]); a[21] = bitselect(a[21]^a[23],a[21],a[22]); a[22] = bitselect(a[22]^a[24],a[22],a[23]); a[23] = bitselect(a[23]^m5,a[23],a[24]); a[24] = bitselect(a[24]^m6,a[24],m5);\
            } \
        } \
    } \
 } while(0)


#define KECCAK_PROCESS(st, in_size, out_size)    do { \
    for (int r = 0; r < 24; ++r) { \
        int os = (r < 23 ? 25 : (out_size));\
        KECCAKF_1600_RND(st, r, os); \
    } \
} while(0)


#define fnv(x, y)        ((x) * FNV_PRIME ^ (y))
#define fnv_reduce(v)    fnv(fnv(fnv(v.x, v.y), v.z), v.w)

typedef union {
    uint uints[128 / sizeof(uint)];
    ulong ulongs[128 / sizeof(ulong)];
    uint2 uint2s[128 / sizeof(uint2)];
    uint4 uint4s[128 / sizeof(uint4)];
    uint8 uint8s[128 / sizeof(uint8)];
    uint16 uint16s[128 / sizeof(uint16)];
    ulong8 ulong8s[128 / sizeof(ulong8)];
} hash128_t;


typedef union {
    ulong8 ulong8s[1];
    ulong4 ulong4s[2];
    uint2 uint2s[8];
    uint4 uint4s[4];
    uint8 uint8s[2];
    uint16 uint16s[1];
    ulong ulongs[8];
    uint  uints[16];
} compute_hash_share;


#ifdef LEGACY

#define MIX(x) \
do { \
    if (get_local_id(0) == lane_idx) { \
        buffer[hash_id] = fnv(init0 ^ (a + x), ((uint *)&mix)[x]) % dag_size; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint idx = buffer[hash_id]; \
    __global hash128_t const* g_dag; \
    g_dag = (__global hash128_t const*) _g_dag0; \
    if (idx & 1) \
        g_dag = (__global hash128_t const*) _g_dag1; \
    mix = fnv(mix, g_dag[idx >> 1].uint8s[thread_id]); \
} while(0)

#else

#define MIX(x) \
do { \
    buffer[get_local_id(0)] = fnv(init0 ^ (a + x), ((uint *)&mix)[x]) % dag_size; \
    uint idx = buffer[lane_idx]; \
    __global hash128_t const* g_dag; \
    g_dag = (__global hash128_t const*) _g_dag0; \
    if (idx & 1) \
        g_dag = (__global hash128_t const*) _g_dag1; \
    mix = fnv(mix, g_dag[idx >> 1].uint8s[thread_id]); \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
} while(0)
#endif

// NOTE: This struct must match the one defined in CLMiner.cpp
struct SearchResults {
    struct {
        uint gid;
        uint mix[8];
        uint pad[7]; // pad to 16 words for easy indexing
    } rslt[MAX_OUTPUTS];
    uint count;
    uint hashCount;
    uint abort;
};

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(
    __global volatile struct SearchResults* restrict g_output,
    __constant uint2 const* g_header,
    __global ulong8 const* _g_dag0,
    __global ulong8 const* _g_dag1,
    uint dag_size,
    ulong start_nonce,
    ulong target
)
{
#ifdef FAST_EXIT
    if (g_output->abort)
        return;
#endif

    const uint thread_id = get_local_id(0) % 4;
    const uint hash_id = get_local_id(0) / 4;
    const uint gid = get_global_id(0);

    __local compute_hash_share sharebuf[WORKSIZE / 4];
#ifdef LEGACY
    __local uint buffer[WORKSIZE / 4];
#else
    __local uint buffer[WORKSIZE];
#endif
    __local compute_hash_share * const share = sharebuf + hash_id;

    // sha3_512(header .. nonce)
    uint2 state[25];
    state[0] = g_header[0];
    state[1] = g_header[1];
    state[2] = g_header[2];
    state[3] = g_header[3];
    state[4] = as_uint2(start_nonce + gid);
    state[5] = as_uint2(0x0000000000000001UL);
    state[6] = (uint2)(0);
    state[7] = (uint2)(0);
    state[8] = as_uint2(0x8000000000000000UL);
    state[9] = (uint2)(0);
    state[10] = (uint2)(0);
    state[11] = (uint2)(0);
    state[12] = (uint2)(0);
    state[13] = (uint2)(0);
    state[14] = (uint2)(0);
    state[15] = (uint2)(0);
    state[16] = (uint2)(0);
    state[17] = (uint2)(0);
    state[18] = (uint2)(0);
    state[19] = (uint2)(0);
    state[20] = (uint2)(0);
    state[21] = (uint2)(0);
    state[22] = (uint2)(0);
    state[23] = (uint2)(0);
    state[24] = (uint2)(0);

    uint2 mixhash[4];

    for (int pass = 0; pass < 2; ++pass) {
        KECCAK_PROCESS(state, select(5, 12, pass != 0), select(8, 1, pass != 0));
        if (pass > 0)
            break;

        uint init0;
        uint8 mix;

#pragma unroll 1
        for (uint tid = 0; tid < 4; tid++) {
            if (tid == thread_id) {
                share->uint2s[0] = state[0];
                share->uint2s[1] = state[1];
                share->uint2s[2] = state[2];
                share->uint2s[3] = state[3];
                share->uint2s[4] = state[4];
                share->uint2s[5] = state[5];
                share->uint2s[6] = state[6];
                share->uint2s[7] = state[7];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            mix = share->uint8s[thread_id & 1];
            init0 = share->uints[0];

            barrier(CLK_LOCAL_MEM_FENCE);

#ifndef LEGACY
#pragma unroll 1
#endif
            for (uint a = 0; a < ACCESSES; a += 8) {
                const uint lane_idx = 4 * hash_id + a / 8 % 4;
                for (uint x = 0; x < 8; ++x)
                    MIX(x);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            share->uint2s[thread_id] = (uint2)(fnv_reduce(mix.lo), fnv_reduce(mix.hi));

            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid == thread_id) {
                state[8] = share->uint2s[0];
                state[9] = share->uint2s[1];
                state[10] = share->uint2s[2];
                state[11] = share->uint2s[3];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        mixhash[0] = state[8];
        mixhash[1] = state[9];
        mixhash[2] = state[10];
        mixhash[3] = state[11];

        state[12] = as_uint2(0x0000000000000001UL);
        state[13] = (uint2)(0);
        state[14] = (uint2)(0);
        state[15] = (uint2)(0);
        state[16] = as_uint2(0x8000000000000000UL);
        state[17] = (uint2)(0);
        state[18] = (uint2)(0);
        state[19] = (uint2)(0);
        state[20] = (uint2)(0);
        state[21] = (uint2)(0);
        state[22] = (uint2)(0);
        state[23] = (uint2)(0);
        state[24] = (uint2)(0);
    }

#ifdef FAST_EXIT
    if (get_local_id(0) == 0)
        atomic_inc(&g_output->hashCount);
#endif

    if (as_ulong(as_uchar8(state[0]).s76543210) <= target) {
#ifdef FAST_EXIT
        atomic_inc(&g_output->abort);
#endif
        uint slot = min(MAX_OUTPUTS - 1u, atomic_inc(&g_output->count));
        g_output->rslt[slot].gid = gid;
        g_output->rslt[slot].mix[0] = mixhash[0].s0;
        g_output->rslt[slot].mix[1] = mixhash[0].s1;
        g_output->rslt[slot].mix[2] = mixhash[1].s0;
        g_output->rslt[slot].mix[3] = mixhash[1].s1;
        g_output->rslt[slot].mix[4] = mixhash[2].s0;
        g_output->rslt[slot].mix[5] = mixhash[2].s1;
        g_output->rslt[slot].mix[6] = mixhash[3].s0;
        g_output->rslt[slot].mix[7] = mixhash[3].s1;
    }
}

typedef union _Node {
    uint dwords[16];
    uint2 qwords[8];
    uint4 dqwords[4];
} Node;

static void keccak_ash(uint2 *dst);
static void SHA3_512(uint2 *s)
{
    uint2 st[25];

    for (uint i = 0; i < 8; ++i)
        st[i] = s[i];

    st[8] = (uint2)(0x00000001, 0x80000000);

    for (uint i = 9; i != 25; ++i)
        st[i] = (uint2)(0);

    KECCAK_PROCESS(st, 8, 8);

    for (uint i = 0; i < 8; ++i)
        s[i] = st[i];
}

__kernel void GenerateDAG(uint start, __global const uint16 *_Cache, __global uint16 *_DAG0, __global uint16 *_DAG1, uint light_size)
{
    __global const Node *Cache = (__global const Node *) _Cache;
    const uint gid = get_global_id(0);
    uint NodeIdx = start + gid;
    const uint thread_id = gid & 3;

    __local Node sharebuf[WORKSIZE];
    __local uint indexbuf[WORKSIZE];
    __local Node *dagNode = sharebuf + (get_local_id(0) / 4) * 4;
    __local uint *indexes = indexbuf + (get_local_id(0) / 4) * 4;
    __global const Node *parentNode;

    Node DAGNode = Cache[NodeIdx % light_size];

    DAGNode.dwords[0] ^= NodeIdx;
    SHA3_512(DAGNode.qwords);

    dagNode[thread_id] = DAGNode;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < 256; ++i) {
        uint ParentIdx = fnv(NodeIdx ^ i, dagNode[thread_id].dwords[i & 15]) % light_size;
        indexes[thread_id] = ParentIdx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint t = 0; t < 4; ++t) {
            uint parentIndex = indexes[t];
            parentNode = Cache + parentIndex;

            dagNode[t].dqwords[thread_id] = fnv(dagNode[t].dqwords[thread_id], parentNode->dqwords[thread_id]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    DAGNode = dagNode[thread_id];

    SHA3_512(DAGNode.qwords);

    __global Node *DAG;
    if (NodeIdx & 2)
        DAG = (__global Node *) _DAG1;
    else
        DAG = (__global Node *) _DAG0;
    NodeIdx &= ~2;
    //if (NodeIdx < DAG_SIZE)
    DAG[(NodeIdx / 2) | (NodeIdx & 1)] = DAGNode;
}

//frk

struct SearchResults_frk
{
    uint count;
    uint hashCount;
    volatile uint abort;
    uint gid[MAX_OUTPUTS];
    ulong sol_targ;
    ulong sol_hea;
};


//output             = arg 0
//header             = arg 1
//start_nonce        = arg 2
//target (boundary)  = arg 3

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1))) __kernel void search_frk(
    __global struct SearchResults_frk* g_output,
    __constant uint2 const* g_header,
    ulong start_nonce,
    ulong target)
{
    if (g_output->abort)
        return;

    const uint thread_id = get_local_id(0) % 4;
    const uint hash_id = get_local_id(0) / 4;
    const uint gid = get_global_id(0);

    __local compute_hash_share sharebuf[WORKSIZE / 4];
    __local uint buffer[WORKSIZE];
    __local compute_hash_share* const share = sharebuf + hash_id;

    // sha3_512(header .. nonce)
    uint2 state[25];
    state[0] = g_header[0];
    state[1] = g_header[1];
    state[2] = g_header[2];
    state[3] = g_header[3];
    state[4] = as_uint2(start_nonce + gid);
    state[5] = as_uint2(0x0000000000000001UL);
    state[6] = (uint2)(0);
    state[7] = (uint2)(0);
    state[8] = as_uint2(0x8000000000000000UL);
    state[9] = (uint2)(0);
    state[10] = (uint2)(0);
    state[11] = (uint2)(0);
    state[12] = (uint2)(0);
    state[13] = (uint2)(0);
    state[14] = (uint2)(0);
    state[15] = (uint2)(0);
    state[16] = (uint2)(0);
    state[17] = (uint2)(0);
    state[18] = (uint2)(0);
    state[19] = (uint2)(0);
    state[20] = (uint2)(0);
    state[21] = (uint2)(0);
    state[22] = (uint2)(0);
    state[23] = (uint2)(0);
    state[24] = (uint2)(0);
    
    // Process Keccak-512
    KECCAK_PROCESS(state, 25, 8);
    //keccak_ash(state);

    state[8] = as_uint2(0x0000000000000001UL);
    state[9] = (uint2)(0);
    state[10] = (uint2)(0);
    state[11] = (uint2)(0);    
    state[12] = (uint2)(0);
    state[13] = (uint2)(0);
    state[14] = (uint2)(0);
    state[15] = (uint2)(0);
    state[16] = as_uint2(0x8000000000000000UL);
    state[17] = (uint2)(0);
    state[18] = (uint2)(0);
    state[19] = (uint2)(0);
    state[20] = (uint2)(0);
    state[21] = (uint2)(0);
    state[22] = (uint2)(0);
    state[23] = (uint2)(0);
    state[24] = (uint2)(0);

    // Process Keccak-256
    KECCAK_PROCESS(state, 25, 8);
    //keccak_ash(state);
   
    if (get_local_id(0) == 0)
        atomic_inc(&g_output->hashCount);
    	
    if (as_ulong(as_uchar8(state[0]).s76543210) <= target)
    {
        atomic_inc(&g_output->abort);
        uint slot = min(MAX_OUTPUTS - 1u, atomic_inc(&g_output->count));
        g_output->gid[slot] = gid;
	g_output->sol_targ = target;
	g_output->sol_hea = as_long(as_uchar8(state[0]).s76543210);
    }
}
//
//
//
//
//#include "common.cl"
#ifndef COMMON_CL_
#define COMMON_CL_

#ifndef __ENDIAN_LITTLE__
#error Your device is not little endian.  Only little endian devices are supported at this time.
#endif

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable


#define UINT_BYTE0(x)   ((uchar)((x) >> 24))
#define UINT_BYTE1(x)   ((uchar)((x) >> 16))
#define UINT_BYTE2(x)   ((uchar)((x) >>  8))
#define UINT_BYTE3(x)   ((uchar)((x)      ))

#define ULONG_BYTE0(x)  ((ulong)((x) >> 56))
#define ULONG_BYTE1(x)  ((ulong)((x) >> 48))
#define ULONG_BYTE2(x)  ((ulong)((x) >> 40))
#define ULONG_BYTE3(x)  ((ulong)((x) >> 32))
#define ULONG_BYTE4(x)  ((ulong)((x) >> 24))
#define ULONG_BYTE5(x)  ((ulong)((x) >> 16))
#define ULONG_BYTE6(x)  ((ulong)((x) >>  8))
#define ULONG_BYTE7(x)  ((ulong)((x)      ))

uint4  MAKE_UINT4( uint a,  uint b,  uint c,  uint d) {  uint4 temp = ( (uint4)(a,b,c,d)); return temp; }
ulong4 MAKE_ULONG4(ulong a, ulong b, ulong c, ulong d) { ulong4 temp = ((ulong4)(a,b,c,d)); return temp; }
uchar4 MAKE_UCHAR4(uchar a, uchar b, uchar c, uchar d) { uchar4 temp = ((uchar4)(a,b,c,d)); return temp; }

#define SPH_C32(x)  ((uint)(x))
#define SPH_T32(x)  SPH_C32(x)
#define SPH_C64(x)  ((ulong)(x))
#define SPH_T64(x)  SPH_C64(x)

#define SPH_ROTL64(x, n)    rotate((ulong)(x), (ulong)(n))
#define SPH_ROTR64(x, n)    SPH_ROTL64(x, (64 - (n)))
#define SWAP32(x)   (as_uint(as_uchar4(x).s3210))
#define SWAP64(x)   (as_ulong(as_uchar8(x).s76543210))

#endif


#define TH_ELT(t, c0, c1, c2, c3, c4, d0, d1, d2, d3, d4) \
{ \
    t = rotate((ulong)(d0 ^ d1 ^ d2 ^ d3 ^ d4), (ulong)1) \
                    ^ (c0 ^ c1 ^ c2 ^ c3 ^ c4);           \
}

#define THETA(b00, b01, b02, b03, b04, \
              b10, b11, b12, b13, b14, \
              b20, b21, b22, b23, b24, \
              b30, b31, b32, b33, b34, \
              b40, b41, b42, b43, b44) \
{ \
    TH_ELT(t0, b40, b41, b42, b43, b44, b10, b11, b12, b13, b14); \
    TH_ELT(t1, b00, b01, b02, b03, b04, b20, b21, b22, b23, b24); \
    TH_ELT(t2, b10, b11, b12, b13, b14, b30, b31, b32, b33, b34); \
    TH_ELT(t3, b20, b21, b22, b23, b24, b40, b41, b42, b43, b44); \
    TH_ELT(t4, b30, b31, b32, b33, b34, b00, b01, b02, b03, b04); \
    b00 ^= t0; b01 ^= t0; b02 ^= t0; b03 ^= t0; b04 ^= t0; \
    b10 ^= t1; b11 ^= t1; b12 ^= t1; b13 ^= t1; b14 ^= t1; \
    b20 ^= t2; b21 ^= t2; b22 ^= t2; b23 ^= t2; b24 ^= t2; \
    b30 ^= t3; b31 ^= t3; b32 ^= t3; b33 ^= t3; b34 ^= t3; \
    b40 ^= t4; b41 ^= t4; b42 ^= t4; b43 ^= t4; b44 ^= t4; \
}

#define RHO(b00, b01, b02, b03, b04, \
            b10, b11, b12, b13, b14, \
            b20, b21, b22, b23, b24, \
            b30, b31, b32, b33, b34, \
            b40, b41, b42, b43, b44) \
{ \
    b01 = rotate(b01, (ulong)36); \
    b02 = rotate(b02, (ulong) 3); \
    b03 = rotate(b03, (ulong)41); \
    b04 = rotate(b04, (ulong)18); \
    b10 = rotate(b10, (ulong) 1); \
    b11 = rotate(b11, (ulong)44); \
    b12 = rotate(b12, (ulong)10); \
    b13 = rotate(b13, (ulong)45); \
    b14 = rotate(b14, (ulong) 2); \
    b20 = rotate(b20, (ulong)62); \
    b21 = rotate(b21, (ulong) 6); \
    b22 = rotate(b22, (ulong)43); \
    b23 = rotate(b23, (ulong)15); \
    b24 = rotate(b24, (ulong)61); \
    b30 = rotate(b30, (ulong)28); \
    b31 = rotate(b31, (ulong)55); \
    b32 = rotate(b32, (ulong)25); \
    b33 = rotate(b33, (ulong)21); \
    b34 = rotate(b34, (ulong)56); \
    b40 = rotate(b40, (ulong)27); \
    b41 = rotate(b41, (ulong)20); \
    b42 = rotate(b42, (ulong)39); \
    b43 = rotate(b43, (ulong) 8); \
    b44 = rotate(b44, (ulong)14); \
}

/*
 * The KHI macro integrates the "lane complement" optimization. On input,
 * some words are complemented:
 *    a00 a01 a02 a04 a13 a20 a21 a22 a30 a33 a34 a43
 * On output, the following words are complemented:
 *    a04 a10 a20 a22 a23 a31
 *
 * The (implicit) permutation and the theta expansion will bring back
 * the input mask for the next round.
 */

#define KHI(b00, b01, b02, b03, b04, \
            b10, b11, b12, b13, b14, \
            b20, b21, b22, b23, b24, \
            b30, b31, b32, b33, b34, \
            b40, b41, b42, b43, b44) \
{ \
    t0 = b00 ^ ( b10 |  b20); \
    t1 = b10 ^ (~b20 |  b30); \
    t2 = b20 ^ ( b30 &  b40); \
    t3 = b30 ^ ( b40 |  b00); \
    t4 = b40 ^ ( b00 &  b10); \
    b00 = t0; b10 = t1; b20 = t2; b30 = t3; b40 = t4; \
    \
    t0 = b01 ^ ( b11 |  b21); \
    t1 = b11 ^ ( b21 &  b31); \
    t2 = b21 ^ ( b31 | ~b41); \
    t3 = b31 ^ ( b41 |  b01); \
    t4 = b41 ^ ( b01 &  b11); \
    b01 = t0; b11 = t1; b21 = t2; b31 = t3; b41 = t4; \
    \
    t0 = b02 ^ ( b12 |  b22); \
    t1 = b12 ^ ( b22 &  b32); \
    t2 = b22 ^ (~b32 &  b42); \
    t3 =~b32 ^ ( b42 |  b02); \
    t4 = b42 ^ ( b02 &  b12); \
    b02 = t0; b12 = t1; b22 = t2; b32 = t3; b42 = t4; \
    \
    t0 = b03 ^ ( b13 &  b23); \
    t1 = b13 ^ ( b23 |  b33); \
    t2 = b23 ^ (~b33 |  b43); \
    t3 =~b33 ^ ( b43 &  b03); \
    t4 = b43 ^ ( b03 |  b13); \
    b03 = t0; b13 = t1; b23 = t2; b33 = t3; b43 = t4; \
    \
    t0 = b04 ^ (~b14 &  b24); \
    t1 =~b14 ^ ( b24 |  b34); \
    t2 = b24 ^ ( b34 &  b44); \
    t3 = b34 ^ ( b44 |  b04); \
    t4 = b44 ^ ( b04 &  b14); \
    b04 = t0; b14 = t1; b24 = t2; b34 = t3; b44 = t4; \
}

#define IOTA(r) { a00 ^= r; }


//void keccak(constant ulong *_wide, constant uint *_buf, uint nonce, ulong *dst)
static void keccak_ash(uint2 *dst)
{
    // Keccak init (doesn't do anything anymore)
    
   
    // Keccak core
    // DECL_STATE
    ulong a00, a01, a02, a03, a04;
    ulong a10, a11, a12, a13, a14;
    ulong a20, a21, a22, a23, a24;
    ulong a30, a31, a32, a33, a34;
    ulong a40, a41, a42, a43, a44;
 
 /*
 //don't read like that --ash   
    // READ_STATE
    a00 = _wide[ 0] ^ ( ((ulong)nonce << 32) | _buf[0] );  
    a10 = _wide[ 1] ^ 0x01;
    a20 = _wide[ 2];
    a30 = _wide[ 3];
    a40 = _wide[ 4];
    a01 = _wide[ 5];
    a11 = _wide[ 6];
    a21 = _wide[ 7];
    a31 = _wide[ 8] ^ 0x8000000000000000;
    a41 = _wide[ 9];
    a02 = _wide[10];
    a12 = _wide[11];
    a22 = _wide[12];
    a32 = _wide[13];
    a42 = _wide[14];
    a03 = _wide[15];
    a13 = _wide[16];
    a23 = _wide[17];
    a33 = _wide[18];
    a43 = _wide[19];
    a04 = _wide[20];
    a14 = _wide[21];
    a24 = _wide[22];
    a34 = _wide[23];
    a44 = _wide[24];
    
    // INPUT_BUF72 (doesn't do anything anymore)
*/
    // Temp variables for THETA and KHI
    ulong t0, t1, t2, t3, t4;

    // i = 0
    THETA ( a00, a01, a02, a03, a04, a10, a11, a12, a13, a14, a20, a21, a22, a23, a24, a30, a31, a32, a33, a34, a40, a41, a42, a43, a44 );
      RHO ( a00, a01, a02, a03, a04, a10, a11, a12, a13, a14, a20, a21, a22, a23, a24, a30, a31, a32, a33, a34, a40, a41, a42, a43, a44 );
      KHI ( a00, a30, a10, a40, a20, a11, a41, a21, a01, a31, a22, a02, a32, a12, a42, a33, a13, a43, a23, a03, a44, a24, a04, a34, a14 );
    IOTA(0x0000000000000001);

    // i = 1
    THETA ( a00, a30, a10, a40, a20, a11, a41, a21, a01, a31, a22, a02, a32, a12, a42, a33, a13, a43, a23, a03, a44, a24, a04, a34, a14 );
      RHO ( a00, a30, a10, a40, a20, a11, a41, a21, a01, a31, a22, a02, a32, a12, a42, a33, a13, a43, a23, a03, a44, a24, a04, a34, a14 );
      KHI ( a00, a33, a11, a44, a22, a41, a24, a02, a30, a13, a32, a10, a43, a21, a04, a23, a01, a34, a12, a40, a14, a42, a20, a03, a31 );
    IOTA(0x0000000000008082);

    // i = 2
    THETA ( a00, a33, a11, a44, a22, a41, a24, a02, a30, a13, a32, a10, a43, a21, a04, a23, a01, a34, a12, a40, a14, a42, a20, a03, a31 );
      RHO ( a00, a33, a11, a44, a22, a41, a24, a02, a30, a13, a32, a10, a43, a21, a04, a23, a01, a34, a12, a40, a14, a42, a20, a03, a31 );
      KHI ( a00, a23, a41, a14, a32, a24, a42, a10, a33, a01, a43, a11, a34, a02, a20, a12, a30, a03, a21, a44, a31, a04, a22, a40, a13 );
    IOTA(0x800000000000808A);

    // i = 3
    THETA ( a00, a23, a41, a14, a32, a24, a42, a10, a33, a01, a43, a11, a34, a02, a20, a12, a30, a03, a21, a44, a31, a04, a22, a40, a13 );
      RHO ( a00, a23, a41, a14, a32, a24, a42, a10, a33, a01, a43, a11, a34, a02, a20, a12, a30, a03, a21, a44, a31, a04, a22, a40, a13 );
      KHI ( a00, a12, a24, a31, a43, a42, a04, a11, a23, a30, a34, a41, a03, a10, a22, a21, a33, a40, a02, a14, a13, a20, a32, a44, a01 );
    IOTA(0x8000000080008000);

    // i = 4
    THETA ( a00, a12, a24, a31, a43, a42, a04, a11, a23, a30, a34, a41, a03, a10, a22, a21, a33, a40, a02, a14, a13, a20, a32, a44, a01 );
      RHO ( a00, a12, a24, a31, a43, a42, a04, a11, a23, a30, a34, a41, a03, a10, a22, a21, a33, a40, a02, a14, a13, a20, a32, a44, a01 );
      KHI ( a00, a21, a42, a13, a34, a04, a20, a41, a12, a33, a03, a24, a40, a11, a32, a02, a23, a44, a10, a31, a01, a22, a43, a14, a30 );
    IOTA(0x000000000000808B);

    // i = 5
    THETA ( a00, a21, a42, a13, a34, a04, a20, a41, a12, a33, a03, a24, a40, a11, a32, a02, a23, a44, a10, a31, a01, a22, a43, a14, a30 );
      RHO ( a00, a21, a42, a13, a34, a04, a20, a41, a12, a33, a03, a24, a40, a11, a32, a02, a23, a44, a10, a31, a01, a22, a43, a14, a30 );
      KHI ( a00, a02, a04, a01, a03, a20, a22, a24, a21, a23, a40, a42, a44, a41, a43, a10, a12, a14, a11, a13, a30, a32, a34, a31, a33 );
    IOTA(0x0000000080000001);

    // i = 6
    THETA ( a00, a02, a04, a01, a03, a20, a22, a24, a21, a23, a40, a42, a44, a41, a43, a10, a12, a14, a11, a13, a30, a32, a34, a31, a33 );
      RHO ( a00, a02, a04, a01, a03, a20, a22, a24, a21, a23, a40, a42, a44, a41, a43, a10, a12, a14, a11, a13, a30, a32, a34, a31, a33 );
      KHI ( a00, a10, a20, a30, a40, a22, a32, a42, a02, a12, a44, a04, a14, a24, a34, a11, a21, a31, a41, a01, a33, a43, a03, a13, a23 );
    IOTA(0x8000000080008081);

    // i = 7
    THETA ( a00, a10, a20, a30, a40, a22, a32, a42, a02, a12, a44, a04, a14, a24, a34, a11, a21, a31, a41, a01, a33, a43, a03, a13, a23 );
      RHO ( a00, a10, a20, a30, a40, a22, a32, a42, a02, a12, a44, a04, a14, a24, a34, a11, a21, a31, a41, a01, a33, a43, a03, a13, a23 );
      KHI ( a00, a11, a22, a33, a44, a32, a43, a04, a10, a21, a14, a20, a31, a42, a03, a41, a02, a13, a24, a30, a23, a34, a40, a01, a12 );
    IOTA(0x8000000000008009);

    // i = 8
    THETA ( a00, a11, a22, a33, a44, a32, a43, a04, a10, a21, a14, a20, a31, a42, a03, a41, a02, a13, a24, a30, a23, a34, a40, a01, a12 );
      RHO ( a00, a11, a22, a33, a44, a32, a43, a04, a10, a21, a14, a20, a31, a42, a03, a41, a02, a13, a24, a30, a23, a34, a40, a01, a12 );
      KHI ( a00, a41, a32, a23, a14, a43, a34, a20, a11, a02, a31, a22, a13, a04, a40, a24, a10, a01, a42, a33, a12, a03, a44, a30, a21 );
    IOTA(0x000000000000008A);

    // i = 9
    THETA ( a00, a41, a32, a23, a14, a43, a34, a20, a11, a02, a31, a22, a13, a04, a40, a24, a10, a01, a42, a33, a12, a03, a44, a30, a21 );
      RHO ( a00, a41, a32, a23, a14, a43, a34, a20, a11, a02, a31, a22, a13, a04, a40, a24, a10, a01, a42, a33, a12, a03, a44, a30, a21 );
      KHI ( a00, a24, a43, a12, a31, a34, a03, a22, a41, a10, a13, a32, a01, a20, a44, a42, a11, a30, a04, a23, a21, a40, a14, a33, a02 );
    IOTA(0x0000000000000088);

    // i = 10
    THETA ( a00, a24, a43, a12, a31, a34, a03, a22, a41, a10, a13, a32, a01, a20, a44, a42, a11, a30, a04, a23, a21, a40, a14, a33, a02 );
      RHO ( a00, a24, a43, a12, a31, a34, a03, a22, a41, a10, a13, a32, a01, a20, a44, a42, a11, a30, a04, a23, a21, a40, a14, a33, a02 );
      KHI ( a00, a42, a34, a21, a13, a03, a40, a32, a24, a11, a01, a43, a30, a22, a14, a04, a41, a33, a20, a12, a02, a44, a31, a23, a10 );
    IOTA(0x0000000080008009);

    // i = 11
    THETA ( a00, a42, a34, a21, a13, a03, a40, a32, a24, a11, a01, a43, a30, a22, a14, a04, a41, a33, a20, a12, a02, a44, a31, a23, a10 );
      RHO ( a00, a42, a34, a21, a13, a03, a40, a32, a24, a11, a01, a43, a30, a22, a14, a04, a41, a33, a20, a12, a02, a44, a31, a23, a10 );
      KHI ( a00, a04, a03, a02, a01, a40, a44, a43, a42, a41, a30, a34, a33, a32, a31, a20, a24, a23, a22, a21, a10, a14, a13, a12, a11 );
    IOTA(0x000000008000000A);

    // i = 12
    THETA ( a00, a04, a03, a02, a01, a40, a44, a43, a42, a41, a30, a34, a33, a32, a31, a20, a24, a23, a22, a21, a10, a14, a13, a12, a11 );
      RHO ( a00, a04, a03, a02, a01, a40, a44, a43, a42, a41, a30, a34, a33, a32, a31, a20, a24, a23, a22, a21, a10, a14, a13, a12, a11 );
      KHI ( a00, a20, a40, a10, a30, a44, a14, a34, a04, a24, a33, a03, a23, a43, a13, a22, a42, a12, a32, a02, a11, a31, a01, a21, a41 );
    IOTA(0x000000008000808B);

    // i = 13
    THETA ( a00, a20, a40, a10, a30, a44, a14, a34, a04, a24, a33, a03, a23, a43, a13, a22, a42, a12, a32, a02, a11, a31, a01, a21, a41 );
      RHO ( a00, a20, a40, a10, a30, a44, a14, a34, a04, a24, a33, a03, a23, a43, a13, a22, a42, a12, a32, a02, a11, a31, a01, a21, a41 );
      KHI ( a00, a22, a44, a11, a33, a14, a31, a03, a20, a42, a23, a40, a12, a34, a01, a32, a04, a21, a43, a10, a41, a13, a30, a02, a24 );
    IOTA(0x800000000000008B);

    // i = 14
    THETA ( a00, a22, a44, a11, a33, a14, a31, a03, a20, a42, a23, a40, a12, a34, a01, a32, a04, a21, a43, a10, a41, a13, a30, a02, a24 );
      RHO ( a00, a22, a44, a11, a33, a14, a31, a03, a20, a42, a23, a40, a12, a34, a01, a32, a04, a21, a43, a10, a41, a13, a30, a02, a24 );
      KHI ( a00, a32, a14, a41, a23, a31, a13, a40, a22, a04, a12, a44, a21, a03, a30, a43, a20, a02, a34, a11, a24, a01, a33, a10, a42 );
    IOTA(0x8000000000008089);

    // i = 15
    THETA ( a00, a32, a14, a41, a23, a31, a13, a40, a22, a04, a12, a44, a21, a03, a30, a43, a20, a02, a34, a11, a24, a01, a33, a10, a42 );
      RHO ( a00, a32, a14, a41, a23, a31, a13, a40, a22, a04, a12, a44, a21, a03, a30, a43, a20, a02, a34, a11, a24, a01, a33, a10, a42 );
      KHI ( a00, a43, a31, a24, a12, a13, a01, a44, a32, a20, a21, a14, a02, a40, a33, a34, a22, a10, a03, a41, a42, a30, a23, a11, a04 );
    IOTA(0x8000000000008003);

    // i = 16
    THETA ( a00, a43, a31, a24, a12, a13, a01, a44, a32, a20, a21, a14, a02, a40, a33, a34, a22, a10, a03, a41, a42, a30, a23, a11, a04 );
      RHO ( a00, a43, a31, a24, a12, a13, a01, a44, a32, a20, a21, a14, a02, a40, a33, a34, a22, a10, a03, a41, a42, a30, a23, a11, a04 );
      KHI ( a00, a34, a13, a42, a21, a01, a30, a14, a43, a22, a02, a31, a10, a44, a23, a03, a32, a11, a40, a24, a04, a33, a12, a41, a20 );
    IOTA(0x8000000000008002);

    // i = 17
    THETA ( a00, a34, a13, a42, a21, a01, a30, a14, a43, a22, a02, a31, a10, a44, a23, a03, a32, a11, a40, a24, a04, a33, a12, a41, a20 );
      RHO ( a00, a34, a13, a42, a21, a01, a30, a14, a43, a22, a02, a31, a10, a44, a23, a03, a32, a11, a40, a24, a04, a33, a12, a41, a20 );
      KHI ( a00, a03, a01, a04, a02, a30, a33, a31, a34, a32, a10, a13, a11, a14, a12, a40, a43, a41, a44, a42, a20, a23, a21, a24, a22 );
    IOTA(0x8000000000000080);

    // i = 18
    THETA ( a00, a03, a01, a04, a02, a30, a33, a31, a34, a32, a10, a13, a11, a14, a12, a40, a43, a41, a44, a42, a20, a23, a21, a24, a22 );
      RHO ( a00, a03, a01, a04, a02, a30, a33, a31, a34, a32, a10, a13, a11, a14, a12, a40, a43, a41, a44, a42, a20, a23, a21, a24, a22 );
      KHI ( a00, a40, a30, a20, a10, a33, a23, a13, a03, a43, a11, a01, a41, a31, a21, a44, a34, a24, a14, a04, a22, a12, a02, a42, a32 );
    IOTA(0x000000000000800A);

    // i = 19
    THETA ( a00, a40, a30, a20, a10, a33, a23, a13, a03, a43, a11, a01, a41, a31, a21, a44, a34, a24, a14, a04, a22, a12, a02, a42, a32 );
      RHO ( a00, a40, a30, a20, a10, a33, a23, a13, a03, a43, a11, a01, a41, a31, a21, a44, a34, a24, a14, a04, a22, a12, a02, a42, a32 );
      KHI ( a00, a44, a33, a22, a11, a23, a12, a01, a40, a34, a41, a30, a24, a13, a02, a14, a03, a42, a31, a20, a32, a21, a10, a04, a43 );
    IOTA(0x800000008000000A);

    // i = 20
    THETA ( a00, a44, a33, a22, a11, a23, a12, a01, a40, a34, a41, a30, a24, a13, a02, a14, a03, a42, a31, a20, a32, a21, a10, a04, a43 );
      RHO ( a00, a44, a33, a22, a11, a23, a12, a01, a40, a34, a41, a30, a24, a13, a02, a14, a03, a42, a31, a20, a32, a21, a10, a04, a43 );
      KHI ( a00, a14, a23, a32, a41, a12, a21, a30, a44, a03, a24, a33, a42, a01, a10, a31, a40, a04, a13, a22, a43, a02, a11, a20, a34 );
    IOTA(0x8000000080008081);

    // i = 21
    THETA ( a00, a14, a23, a32, a41, a12, a21, a30, a44, a03, a24, a33, a42, a01, a10, a31, a40, a04, a13, a22, a43, a02, a11, a20, a34 );
      RHO ( a00, a14, a23, a32, a41, a12, a21, a30, a44, a03, a24, a33, a42, a01, a10, a31, a40, a04, a13, a22, a43, a02, a11, a20, a34 );
      KHI ( a00, a31, a12, a43, a24, a21, a02, a33, a14, a40, a42, a23, a04, a30, a11, a13, a44, a20, a01, a32, a34, a10, a41, a22, a03 );
    IOTA(0x8000000000008080);

    // i = 22
    THETA ( a00, a31, a12, a43, a24, a21, a02, a33, a14, a40, a42, a23, a04, a30, a11, a13, a44, a20, a01, a32, a34, a10, a41, a22, a03 );
      RHO ( a00, a31, a12, a43, a24, a21, a02, a33, a14, a40, a42, a23, a04, a30, a11, a13, a44, a20, a01, a32, a34, a10, a41, a22, a03 );
      KHI ( a00, a13, a21, a34, a42, a02, a10, a23, a31, a44, a04, a12, a20, a33, a41, a01, a14, a22, a30, a43, a03, a11, a24, a32, a40 );
    IOTA(0x0000000080000001);

    // i = 23
    THETA ( a00, a13, a21, a34, a42, a02, a10, a23, a31, a44, a04, a12, a20, a33, a41, a01, a14, a22, a30, a43, a03, a11, a24, a32, a40 );
      RHO ( a00, a13, a21, a34, a42, a02, a10, a23, a31, a44, a04, a12, a20, a33, a41, a01, a14, a22, a30, a43, a03, a11, a24, a32, a40 );
      KHI ( a00, a01, a02, a03, a04, a10, a11, a12, a13, a14, a20, a21, a22, a23, a24, a30, a31, a32, a33, a34, a40, a41, a42, a43, a44 );
    IOTA(0x8000000080008008);

    
    // WRITE_STATE

    dst[0] =  a00;
    dst[1] = ~a10;
    dst[2] = ~a20;
    dst[3] =  a30;
    dst[4] =  a40;
    dst[5] =  a01;
    dst[6] =  a11;
    dst[7] =  a21;

}
