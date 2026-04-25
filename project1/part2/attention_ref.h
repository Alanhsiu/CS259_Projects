#pragma once
#include <math.h>
#include <float.h>
#include <stdlib.h>

static const int ATTN_D = 64;

static void attn_fill(float* m, long long n, int seed)
{
    for (long long i = 0; i < n; i++)
        m[i] = 0.1f * sinf((float)(i * 3 + seed * 7));
}

static void flash_prefill_ref(
    const float Q[][ATTN_D],
    const float K[][ATTN_D],
    const float V[][ATTN_D],
    float       O[][ATTN_D],
    int S)
{
    float inv_sqrt = 1.f / sqrtf((float)ATTN_D);
    for (int i = 0; i < S; i++) {
        float out[ATTN_D] = {};
        float m = -FLT_MAX, d = 0.f;
        for (int j = 0; j <= i; j++) {
            float score = 0.f;
            for (int h = 0; h < ATTN_D; h++)
                score += Q[i][h] * K[j][h];
            score *= inv_sqrt;
            float m_new      = (score > m) ? score : m;
            float correction = expf(m - m_new);
            float exp_score  = expf(score - m_new);
            d = d * correction + exp_score;
            for (int h = 0; h < ATTN_D; h++)
                out[h] = out[h] * correction + exp_score * V[j][h];
            m = m_new;
        }
        for (int h = 0; h < ATTN_D; h++) O[i][h] = out[h] / d;
    }
}

static void standard_decode_ref(
    const float  q[ATTN_D],
    const float  K[][ATTN_D],
    const float  V[][ATTN_D],
    float        o[ATTN_D],
    int C)
{
    float inv_sqrt = 1.f / sqrtf((float)ATTN_D);
    float* scores = (float*)malloc(C * sizeof(float));
    for (int j = 0; j < C; j++) {
        float s = 0.f;
        for (int h = 0; h < ATTN_D; h++) s += q[h] * K[j][h];
        scores[j] = s * inv_sqrt;
    }
    float mx = scores[0];
    for (int j = 1; j < C; j++) if (scores[j] > mx) mx = scores[j];
    float sum = 0.f;
    for (int j = 0; j < C; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
    for (int j = 0; j < C; j++) scores[j] /= sum;
    for (int h = 0; h < ATTN_D; h++) {
        float acc = 0.f;
        for (int j = 0; j < C; j++) acc += scores[j] * V[j][h];
        o[h] = acc;
    }
    free(scores);
}

static float attn_max_diff(const float* a, const float* b, int n)
{
    float md = 0.f;
    for (int i = 0; i < n; i++)
        md = fmaxf(md, fabsf(a[i] - b[i]));
    return md;
}
