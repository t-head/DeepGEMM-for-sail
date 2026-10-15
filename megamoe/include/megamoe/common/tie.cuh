#pragma once

#include <megamoe/common/base.h>

#define DG_TIE_CONCAT_IMPL(A, B) A##B
#define DG_TIE_CONCAT(A, B) DG_TIE_CONCAT_IMPL(A, B)

#define DG_TIE_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define DG_TIE_COUNT_ARGS(...) DG_TIE_GET_NTH_ARG(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define DG_TIE_OP_DECL(I, TUPLE, VAR) auto VAR = ::megamoe::get<I>(TUPLE)

#define DG_TIE_APPLY_OP_1(OP, T, V1) OP(0, T, V1);
#define DG_TIE_APPLY_OP_2(OP, T, V1, V2) \
    OP(0, T, V1);                        \
    OP(1, T, V2);
#define DG_TIE_APPLY_OP_3(OP, T, V1, V2, V3) \
    OP(0, T, V1);                            \
    OP(1, T, V2);                            \
    OP(2, T, V3);
#define DG_TIE_APPLY_OP_4(OP, T, V1, V2, V3, V4) \
    OP(0, T, V1);                                \
    OP(1, T, V2);                                \
    OP(2, T, V3);                                \
    OP(3, T, V4);

#define DG_TIE_DECL(TUPLE_EXPR, ...)                                        \
    auto&& DG_TIE_CONCAT(megamoe_tie_temp_tuple_, __LINE__) = (TUPLE_EXPR); \
    DG_TIE_CONCAT(DG_TIE_APPLY_OP_, DG_TIE_COUNT_ARGS(__VA_ARGS__))         \
    (DG_TIE_OP_DECL, DG_TIE_CONCAT(megamoe_tie_temp_tuple_, __LINE__), __VA_ARGS__)
