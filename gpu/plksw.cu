#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <smmintrin.h>  // for SSE.4.1
#include <cuda_runtime.h>
#include "../ksw2.h"


int ksw_global_scalar(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target, int8_t m, const int8_t *mat, int8_t q, int8_t e, int w, int *m_cigar_, int *n_cigar_, uint32_t **cigar_) {

    int qe = q + e, qe2 = qe + qe;    // gap penalty
    int last_H0_t = 0, H0 = 0; 
    int8_t *u;   // H[i,j] - H[i-1,j]
    int8_t *v;   // H[i,j] - H[i,j-1]
    int8_t *x;   // E[i+1,j] - H[i,j]
    int8_t *y;   // F[i,j+1] - H[i,j]
    int8_t *s;   

    uint8_t *p = 0, *qr;
    int *off = 0;
    u = (int8_t*)kcalloc(km, tlen + 1, 1);
    v = (int8_t*)kcalloc(km, tlen + 1, 1);
    x = (int8_t*)kcalloc(km, tlen + 1, 1);
    y = (int8_t*)kcalloc(km, tlen + 1, 1);
    s = (int8_t*)kmalloc(km, tlen);

    qr = (uint8_t*)kmalloc(km, qlen); // reverse query
    if(w < 0) w = tlen > qlen ? tlen : qlen;
    int n_col = w + 1 < tlen ? w + 1 : tlen;

    if(m_cigar_ && n_cigar_ && cigar_) {
        p = (uint8_t*)kcalloc(km, (size_t)(qlen + tlen)*n_col, 1);
        off = (int*)kmalloc(km, (qlen + tlen) * sizeof(int));
    }

    int t;
    for(t = 0; t < qlen; t ++){
        qr[t] = query[qlen - 1 - t];
    }

    /*
        @preliminary:  r = j + i
        @preliminary:  t = i
        @coor:  [i, j] ——> [r, t]
        @result: get the range of row coor i

            query (j)
        ************
        *
ref(i)  * 
        * 
        *
        * 
    */
    int r;
    for(r = 0; r < qlen + tlen - 1; r++) {
        int st = 0, en = tlen - 1;
        int8_t x1, v1;

        // j ∈ [st, en] satisfies 1. i < qlen;  2. j < r 
        if(st < r - qlen + 1) st = r - qlen + 1;
        if(en > r) en = r;

        // j ∈ [st, en] satisfies |i-j| <= w which equals i ∈ [(r-w)/2, (r+w)/2]
        if(st < (r-w+1)>>1) st = (r - w + 1) >> 1;     // ceil   
        if(en > (r+w)>>1) en = (r + w) >> 1;   // floor

        // get previous difference state: x1/v1  y[en]/u[en] 
        // [r-1, st-1] ∈ band -> st/en - 1 ∈ [(r-1-w)/2, (r-1+w)/2]  -> [st, en] ∈ [2st-1-w, 2st+w-1]
        if(st != 0){
            // st < (r-1-w)/2
            if(r > st + st + w - 1) x1 = 0, v1 = 0;
            else x1 = s[st - 1], v1 = v[st - 1];
        }else x1 = 0, v1 = r ? q : 0;
        if(en != r) {
            if(r < en + en - w - 1) y[en] = u[en] = 0;
        }else y[r] = 0, u[r] = r ? q : 0;

        // match/mismatch score for coor[t, r-t] of matrix
        for(t = st; t <= en; ++t) {
            s[t] = mat[target[t] * m + qr[t + qlen - 1 - r]];
        }

        if(m_cigar_ && n_cigar_ && cigar_) {
            uint8_t *pr = p + (size_t)r * n_col;
            off[r] = st;
            for(t = st; t <= en; t++) {
                 /*
                    At the beginning of the loop:
                        v1 = v(r-1, t-1), x1 = x(r-1, t-1)
                        u[t] = u(r-1, t), v[t] = v(r-1, t)
                        x[t] = x(r-1, t), y[t] = y(r-1, t)

                        a = x(r-1, t-1) + v(r-1, t-1)
                        b = y(r-1, t) + u(r-1, t)
                        z = max{S(t, r-t)+2q+2r, a, b}

                        u(r, t) = z - v(r-1,t-1)
                        v(r,t) = z - u(r-1,t-1)
                        x(r,t) = max{0, a-z+q}
                        y(r,t) = max{0, b-z+q}
                 */
                uint8_t d;
                int8_t u1;
                int8_t z = s[t] + qe2;
                int8_t a = x1 + v1;
                int8_t b = y[t] + u[t];
                d = a > z ? 1 : 0; 
                z = a > z ? a : z;
                d = b > z ? 2 : d;
                z = b > z ? b : z;
                u1 = u[t];
                u[t] = z - v1;
                v1 = v[t];
                v[t] = z - u1;
                z -= q;
                a -= z;
                b -= z;
                x1 = x[t];
                d |= a > 0 ? 0x08 : 0;
                x[t] = a > 0 ? a : 0;
                d |= b > 0 ? 0x10 : 0;
                y[t] = b > 0 ? b : 0;
                pr[t - st] = d;
            }
        }else {
            for(t = st; t <= en; t ++) {
                int8_t u1;
                int8_t z = s[t] + qe2;
                int8_t a = x1 + v1;
                int8_t b = y[t] + u[t];
                z = a > z ? a : z;
                z = b > z ? b : z;
                u1 = u[t];
                u[t] = z - v1;
                v1 = v[t];
                v[t] = z - u1;
                z -= q;
                a -= z;
                b -= z;
                x1 = x[t];
                x[t] = a > 0 ? a : 0;
                y[t] = b > 0 ? b : 0;
            }
        }
        if(r > 0) {
            if(last_H0_t >= st && last_H0_t <= en) {
                H0 += v[last_H0_t] - qe;
            }else ++last_H0_t, H0 += u[last_H0_t] - qe;
        }else H0 = v[0] - qe - qe, last_H0_t = 0;
    }
    kfree(km, u); kfree(km, v); kfree(km, x); kfree(km, y); kfree(km, s); kfree(km, qr);
    if(m_cigar_ && n_cigar_ && cigar_) {
        ksw_backtrack(km, 1, 0, 0, p, off, 0, n_col, tlen-1, qlen-1, m_cigar_, n_cigar_, cigar_);
        kfree(km, p);
        kfree(km, off);
    }

    return H0;
}

int ksw_global_sse_4_1(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target, int8_t m, const int8_t *mat, int8_t q, int8_t e, int w, int *m_cigar_, int *n_cigar_, uint32_t **cigar_) {

    int n_col;
    int last_H0_t = 0, H0 = 0; 

    __m128i *u, *v, *x, *y, *s, *p;
    __m128i q_, qe2_, zero_, flag1_, flag2_, flag8_, flag16_;


    zero_ = _mm_set1_epi8(0);
    q_ = _mm_set1_epi8(q);
    qe2_ = _mm_set1_epi8((q + e) * 2);
    flag1_ = _mm_set1_epi8(1);
    flag2_ = _mm_set1_epi8(2);
    flag8_ = _mm_set1_epi8(0x08);
    flag16_ = _mm_set1_epi8(0x10);

    int tlen_, n_col_;
    if(w < 0) w = tlen > qlen ? tlen : qlen;
    n_col = w + 1 > tlen ? w + 1 : tlen;
    tlen_ = (tlen + 15) / 16;
    n_col_ = (n_col + 15) / 16 + 1;
    n_col = n_col_ * 16;

    int *off;
    uint8_t *qr, *mem, *mem2;
    mem = (uint8_t*)kcalloc(km, tlen_ * 5 + 1, 16);
    u = (__m128i*)(((size_t)mem + 15) >> 4 << 4);
    v = u + tlen_, x = v + tlen_, y = x + tlen_, s = y + tlen_;
    qr = (uint8_t*)kcalloc(km, qlen, 1);
    mem2 = (uint8_t*)kmalloc(km, ((size_t)(qlen + tlen - 1) * n_col_ + 1) * 16);
    p = (__m128i*)(((size_t)mem2 + 15) >> 4 << 4);
    off = (int*)kmalloc(km, (qlen + tlen - 1) * sizeof(int));

    int r, t;
    for(t = 0; t < qlen; t++) {
        qr[t] = query[qlen - 1 - t];
    }

    int last_st, last_en;
    for(r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {

        int st = 0, en = tlen - 1, st0, en0, st_, en_;
        int8_t x1, v1;
        __m128i x1_, v1_, *pr;

        if(st < r - qlen + 1) st = r - qlen + 1;
        if(en > r) en = r;
        if(st < (r-w+1)>>1) st = (r-w+1)>>1;
        if(en > (r+w)>>1) en = (r+w)>>1;
        st0 = st, en0 = en;
        st = st / 16 * 16, en = (en + 16)/16*16-1;
        off[r] = st;
        
        if(st > 0){
            if(st-1 >= last_st && st - 1 <= last_en)
                x1 = ((uint8_t*)x)[st - 1], v1 = ((uint8_t*)v)[st - 1];
            else x1 = v1 = 0;
        }else x1 = 0, v1 = r ? q : 0;

        if(en >= r) ((uint8_t*)y)[r] = 0, ((uint8_t*)u)[r] = r ? q : 0;

        for(t = st0; t <= en0; ++t)
            ((uint8_t*)s)[t] = mat[target[t] * m + qr[t + qlen - 1 - r]];

        x1_ = _mm_cvtsi32_si128(x1);
        v1_ = _mm_cvtsi32_si128(v1);

        st_ = st >> 4, en_ = en >> 4;
        pr = p + (size_t)r * n_col_ - st_;

        for(t = st_; t <= en_; ++t) {
            __m128i d, z, a, b, xt1, vt1, ut, tmp;
            
            z = _mm_add_epi8(_mm_load_si128(&s[t]), qe2_);

            xt1 = _mm_load_si128(&x[t]);
            tmp = _mm_srli_si128(xt1, 15);
            xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_);
            x1_ = tmp;

            vt1 = _mm_load_si128(&v[t]);
            tmp = _mm_srli_si128(vt1, 15);
            vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_);
            v1_ = tmp;
            a = _mm_add_epi8(xt1, vt1);

            ut = _mm_load_si128(&u[t]);
            b = _mm_add_epi8(_mm_load_si128(&y[t]), ut);


            d = _mm_and_si128(_mm_cmpgt_epi8(a, z), flag1_);
            z = _mm_max_epi8(z, a);
            tmp = _mm_cmpgt_epi8(b, z);
            d = _mm_blendv_epi8(d, flag2_, tmp);
            z = _mm_max_epu8(z, b);
            _mm_store_si128(&u[t], _mm_sub_epi8(z, vt1));
            _mm_store_si128(&v[t], _mm_sub_epi8(z, ut));

            z = _mm_sub_epi8(z, q_);
            a = _mm_sub_epi8(a, z);
            b = _mm_sub_epi8(b, z);

            tmp = _mm_cmpgt_epi8(a, zero_);
            d = _mm_or_si128(d, _mm_and_si128(flag8_, tmp));
            _mm_store_si128(&x[t], _mm_and_si128(a, tmp));

            tmp = _mm_cmpgt_epi8(b, zero_);
            d = _mm_or_si128(d, _mm_and_si128(flag16_, tmp));               
            _mm_store_si128(&y[t], _mm_and_si128(b, tmp));


            _mm_store_si128(&pr[t], d);
        }
        if(r > 0) {
            if(last_H0_t >= st0 && last_H0_t <= en0) {
                H0 += ((uint8_t*)v)[last_H0_t] - (q + e);
            }else ++last_H0_t, H0 += ((uint8_t*)u)[last_H0_t] - (q + e);

        }else H0 = ((uint8_t*)v)[0] - 2 * (q + e), last_H0_t = 0;
        last_st = st, last_en = en;
    }
    kfree(km, mem);
    kfree(km, qr);
    ksw_backtrack(km, 1, 0, 0, (uint8_t*)p, off, 0, n_col, tlen-1, qlen-1, m_cigar_, n_cigar_, cigar_);
    kfree(km, mem2);
    kfree(km, off);
    return H0;
}

int ksw_extd2_sse4_1(void *km, int qlen, const uint8_t* query, int tlen, const uint8_t* target, int8_t m, const int8_t* mat,
                        int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, int end_bonus, int flag, ksw_extz_t *ez) {

#define __dp_code_block1 \
	z = _mm_load_si128(&s[t]); \
	xt1 = _mm_load_si128(&x[t]);                     /* xt1 <- x[r-1][t..t+15] */ \
	tmp = _mm_srli_si128(xt1, 15);                   /* tmp <- x[r-1][t+15] */ \
	xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_); /* xt1 <- x[r-1][t-1..t+14] */ \
	x1_ = tmp; \
	vt1 = _mm_load_si128(&v[t]);                     /* vt1 <- v[r-1][t..t+15] */ \
	tmp = _mm_srli_si128(vt1, 15);                   /* tmp <- v[r-1][t+15] */ \
	vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_); /* vt1 <- v[r-1][t-1..t+14] */ \
	v1_ = tmp; \
	a = _mm_add_epi8(xt1, vt1);                      /* a <- x[r-1][t-1..t+14] + v[r-1][t-1..t+14] */ \
	ut = _mm_load_si128(&u[t]);                      /* ut <- u[t..t+15] */ \
	b = _mm_add_epi8(_mm_load_si128(&y[t]), ut);     /* b <- y[r-1][t..t+15] + u[r-1][t..t+15] */ \
	x2t1= _mm_load_si128(&x2[t]); \
	tmp = _mm_srli_si128(x2t1, 15); \
	x2t1= _mm_or_si128(_mm_slli_si128(x2t1, 1), x21_); \
	x21_= tmp; \
	a2= _mm_add_epi8(x2t1, vt1); \
	b2= _mm_add_epi8(_mm_load_si128(&y2[t]), ut);

#define __dp_code_block2 \
	_mm_store_si128(&u[t], _mm_sub_epi8(z, vt1));    /* u[r][t..t+15] <- z - v[r-1][t-1..t+14] */ \
	_mm_store_si128(&v[t], _mm_sub_epi8(z, ut));     /* v[r][t..t+15] <- z - u[r-1][t..t+15] */ \
	tmp = _mm_sub_epi8(z, q_); \
	a = _mm_sub_epi8(a, tmp); \
	b = _mm_sub_epi8(b, tmp); \
	tmp = _mm_sub_epi8(z, q2_); \
	a2= _mm_sub_epi8(a2, tmp); \
	b2= _mm_sub_epi8(b2, tmp);

	int r, t, qe = q + e, n_col_, *off = 0, *off_end = 0, tlen_, qlen_, last_st, last_en, wl, wr, max_sc, min_sc, long_thres, long_diff;
	int with_cigar = !(flag&KSW_EZ_SCORE_ONLY), approx_max = !!(flag&KSW_EZ_APPROX_MAX);
	int32_t *H = 0, H0 = 0, last_H0_t = 0;
	uint8_t *qr, *sf, *mem, *mem2 = 0;
	__m128i q_, q2_, qe_, qe2_, zero_, sc_mch_, sc_mis_, m1_, sc_N_;
	__m128i *u, *v, *x, *y, *x2, *y2, *s, *p = 0;

	ksw_reset_extz(ez);
	if (m <= 1 || qlen <= 0 || tlen <= 0) return;

	if (q2 + e2 < q + e) t = q, q = q2, q2 = t, t = e, e = e2, e2 = t; // make sure q+e no larger than q2+e2

	zero_   = _mm_set1_epi8(0);
	q_      = _mm_set1_epi8(q);
	q2_     = _mm_set1_epi8(q2);
	qe_     = _mm_set1_epi8(q + e);
	qe2_    = _mm_set1_epi8(q2 + e2);
	sc_mch_ = _mm_set1_epi8(mat[0]);
	sc_mis_ = _mm_set1_epi8(mat[1]);
	sc_N_   = mat[m*m-1] == 0? _mm_set1_epi8(-e2) : _mm_set1_epi8(mat[m*m-1]);
	m1_     = _mm_set1_epi8(m - 1); // wildcard

	if (w < 0) w = tlen > qlen? tlen : qlen;
	wl = wr = w;
	tlen_ = (tlen + 15) / 16;
	n_col_ = qlen < tlen? qlen : tlen;
	n_col_ = ((n_col_ < w + 1? n_col_ : w + 1) + 15) / 16 + 1;
	qlen_ = (qlen + 15) / 16;
	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) {
		max_sc = max_sc > mat[t]? max_sc : mat[t];
		min_sc = min_sc < mat[t]? min_sc : mat[t];
	}
	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches

	long_thres = e != e2? (q2 - q) / (e - e2) - 1 : 0;
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	mem = (uint8_t*)kcalloc(km, tlen_ * 8 + qlen_ + 1, 16);
	u = (__m128i*)(((size_t)mem + 15) >> 4 << 4); // 16-byte aligned
	v = u + tlen_, x = v + tlen_, y = x + tlen_, x2 = y + tlen_, y2 = x2 + tlen_;
	s = y2 + tlen_, sf = (uint8_t*)(s + tlen_), qr = sf + tlen_ * 16;
	memset(u,  -q  - e,  tlen_ * 16);
	memset(v,  -q  - e,  tlen_ * 16);
	memset(x,  -q  - e,  tlen_ * 16);
	memset(y,  -q  - e,  tlen_ * 16);
	memset(x2, -q2 - e2, tlen_ * 16);
	memset(y2, -q2 - e2, tlen_ * 16);
	if (!approx_max) {
		H = (int32_t*)kmalloc(km, tlen_ * 16 * 4);
		for (t = 0; t < tlen_ * 16; ++t) H[t] = KSW_NEG_INF;
	}
	if (with_cigar) {
		mem2 = (uint8_t*)kmalloc(km, ((size_t)(qlen + tlen - 1) * n_col_ + 1) * 16);
		p = (__m128i*)(((size_t)mem2 + 15) >> 4 << 4);
		off = (int*)kmalloc(km, (qlen + tlen - 1) * sizeof(int) * 2);
		off_end = off + qlen + tlen - 1;
	}

	for (t = 0; t < qlen; ++t) qr[t] = query[qlen - 1 - t];
	memcpy(sf, target, tlen);

	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		uint8_t *qrr = qr + (qlen - 1 - r);
		int8_t *u8 = (int8_t*)u, *v8 = (int8_t*)v, *x8 = (int8_t*)x, *x28 = (int8_t*)x2;
		__m128i x1_, x21_, v1_;
		// find the boundaries
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r-wr+1)>>1) st = (r-wr+1)>>1; // take the ceil
		if (en > (r+wl)>>1) en = (r+wl)>>1; // take the floor
		if (st > en) {
			ez->zdropped = 1;
			break;
		}
		st0 = st, en0 = en;
		st = st / 16 * 16, en = (en + 16) / 16 * 16 - 1;
		// set boundary conditions
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en) {
				x1 = x8[st - 1], x21 = x28[st - 1], v1 = v8[st - 1]; // (r-1,s-1) calculated in the last round
			} else {
				x1 = -q - e, x21 = -q2 - e2;
				v1 = -q - e;
			}
		} else {
			x1 = -q - e, x21 = -q2 - e2;
			v1 = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		if (en >= r) {
			((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
			u8[r] = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		// loop fission: set scores first
		if (!(flag & KSW_EZ_GENERIC_SC)) {
			for (t = st0; t <= en0; t += 16) {
				__m128i sq, st, tmp, mask;
				sq = _mm_loadu_si128((__m128i*)&sf[t]);
				st = _mm_loadu_si128((__m128i*)&qrr[t]);
				mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(st, m1_));
				tmp = _mm_cmpeq_epi8(sq, st);
				tmp = _mm_blendv_epi8(sc_mis_, sc_mch_, tmp);
				tmp = _mm_blendv_epi8(tmp,     sc_N_,   mask);
    			_mm_storeu_si128((__m128i*)((int8_t*)s + t), tmp);
			}
		} else {
			for (t = st0; t <= en0; ++t)
				((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
		}
		// core loop
		x1_  = _mm_cvtsi32_si128((uint8_t)x1);
		x21_ = _mm_cvtsi32_si128((uint8_t)x21);
		v1_  = _mm_cvtsi32_si128((uint8_t)v1);
		st_ = st / 16, en_ = en / 16;
		assert(en_ - st_ + 1 <= n_col_);
		if (!with_cigar) { // score only
			for (t = st_; t <= en_; ++t) {
				__m128i z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				z = _mm_max_epi8(z, a);
				z = _mm_max_epi8(z, b);
				z = _mm_max_epi8(z, a2);
				z = _mm_max_epi8(z, b2);
				z = _mm_min_epi8(z, sc_mch_);
				__dp_code_block2; // save u[] and v[]; update a, b, a2 and b2
				_mm_store_si128(&x[t],  _mm_sub_epi8(_mm_max_epi8(a,  zero_), qe_));
				_mm_store_si128(&y[t],  _mm_sub_epi8(_mm_max_epi8(b,  zero_), qe_));
				_mm_store_si128(&x2[t], _mm_sub_epi8(_mm_max_epi8(a2, zero_), qe2_));
				_mm_store_si128(&y2[t], _mm_sub_epi8(_mm_max_epi8(b2, zero_), qe2_));
			}
		} else if (!(flag&KSW_EZ_RIGHT)) { // gap left-alignment
			__m128i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m128i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				d = _mm_and_si128(_mm_cmpgt_epi8(a, z), _mm_set1_epi8(1));       // d = a  > z? 1 : 0
				z = _mm_max_epi8(z, a);
				d = _mm_blendv_epi8(d, _mm_set1_epi8(2), _mm_cmpgt_epi8(b,  z)); // d = b  > z? 2 : d
				z = _mm_max_epi8(z, b);
				d = _mm_blendv_epi8(d, _mm_set1_epi8(3), _mm_cmpgt_epi8(a2, z)); // d = a2 > z? 3 : d
				z = _mm_max_epi8(z, a2);
				d = _mm_blendv_epi8(d, _mm_set1_epi8(4), _mm_cmpgt_epi8(b2, z)); // d = a2 > z? 3 : d
				z = _mm_max_epi8(z, b2);
				z = _mm_min_epi8(z, sc_mch_);
				__dp_code_block2;
				tmp = _mm_cmpgt_epi8(a, zero_);
				_mm_store_si128(&x[t],  _mm_sub_epi8(_mm_and_si128(tmp, a),  qe_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x08))); // d = a > 0? 1<<3 : 0
				tmp = _mm_cmpgt_epi8(b, zero_);
				_mm_store_si128(&y[t],  _mm_sub_epi8(_mm_and_si128(tmp, b),  qe_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x10))); // d = b > 0? 1<<4 : 0
				tmp = _mm_cmpgt_epi8(a2, zero_);
				_mm_store_si128(&x2[t], _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x20))); // d = a > 0? 1<<5 : 0
				tmp = _mm_cmpgt_epi8(b2, zero_);
				_mm_store_si128(&y2[t], _mm_sub_epi8(_mm_and_si128(tmp, b2), qe2_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x40))); // d = b > 0? 1<<6 : 0
				_mm_store_si128(&pr[t], d);
			}
		} else { // gap right-alignment
			__m128i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m128i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				d = _mm_andnot_si128(_mm_cmpgt_epi8(z, a), _mm_set1_epi8(1));    // d = z > a?  0 : 1
				z = _mm_max_epi8(z, a);
				d = _mm_blendv_epi8(_mm_set1_epi8(2), d, _mm_cmpgt_epi8(z, b));  // d = z > b?  d : 2
				z = _mm_max_epi8(z, b);
				d = _mm_blendv_epi8(_mm_set1_epi8(3), d, _mm_cmpgt_epi8(z, a2)); // d = z > a2? d : 3
				z = _mm_max_epi8(z, a2);
				d = _mm_blendv_epi8(_mm_set1_epi8(4), d, _mm_cmpgt_epi8(z, b2)); // d = z > b2? d : 4
				z = _mm_max_epi8(z, b2);
				z = _mm_min_epi8(z, sc_mch_);
				__dp_code_block2;
				tmp = _mm_cmpgt_epi8(zero_, a);
				_mm_store_si128(&x[t],  _mm_sub_epi8(_mm_andnot_si128(tmp, a),  qe_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x08))); // d = a > 0? 1<<3 : 0
				tmp = _mm_cmpgt_epi8(zero_, b);
				_mm_store_si128(&y[t],  _mm_sub_epi8(_mm_andnot_si128(tmp, b),  qe_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x10))); // d = b > 0? 1<<4 : 0
				tmp = _mm_cmpgt_epi8(zero_, a2);
				_mm_store_si128(&x2[t], _mm_sub_epi8(_mm_andnot_si128(tmp, a2), qe2_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x20))); // d = a > 0? 1<<5 : 0
				tmp = _mm_cmpgt_epi8(zero_, b2);
				_mm_store_si128(&y2[t], _mm_sub_epi8(_mm_andnot_si128(tmp, b2), qe2_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x40))); // d = b > 0? 1<<6 : 0
				_mm_store_si128(&pr[t], d);
			}
		}
		if (!approx_max) { // find the exact max with a 32-bit score array
			int32_t max_H, max_t;
			// compute H[], max_H and max_t
			if (r > 0) {
				int32_t HH[4], tt[4], en1 = st0 + (en0 - st0) / 4 * 4, i;
				__m128i max_H_, max_t_;
				max_H = H[en0] = en0 > 0? H[en0-1] + u8[en0] : H[en0] + v8[en0]; // special casing the last element
				max_t = en0;
				max_H_ = _mm_set1_epi32(max_H);
				max_t_ = _mm_set1_epi32(max_t);
				for (t = st0; t < en1; t += 4) { // this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t;
					__m128i H1, tmp, t_;
					H1 = _mm_loadu_si128((__m128i*)&H[t]);
					t_ = _mm_setr_epi32(v8[t], v8[t+1], v8[t+2], v8[t+3]);
					H1 = _mm_add_epi32(H1, t_);
					_mm_storeu_si128((__m128i*)&H[t], H1);
					t_ = _mm_set1_epi32(t);
					tmp = _mm_cmpgt_epi32(H1, max_H_);
					max_H_ = _mm_blendv_epi8(max_H_, H1, tmp);
					max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
				}
				_mm_storeu_si128((__m128i*)HH, max_H_);
				_mm_storeu_si128((__m128i*)tt, max_t_);
				for (i = 0; i < 4; ++i)
					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;
				for (; t < en0; ++t) { // for the rest of values that haven't been computed with SSE
					H[t] += (int32_t)v8[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}
			} else H[0] = v8[0] - qe, max_H = H[0], max_t = 0; // special casing r==0
			// update ez
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop(ez, 1, max_H, r, max_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];
		} else { // find approximate max; Z-drop might be inaccurate, too.
			if (r > 0) {
				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v8[last_H0_t];
					int32_t d1 = u8[last_H0_t + 1];
					if (d0 > d1) H0 += d0;
					else H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v8[last_H0_t];
				} else {
					++last_H0_t, H0 += u8[last_H0_t];
				}
			} else H0 = v8[0] - qe, last_H0_t = 0;
			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop(ez, 1, H0, r, last_H0_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}
	kfree(km, mem);
	if (!approx_max) kfree(km, H);
	if (with_cigar) { // backtrack
		int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
		if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY)) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (!ez->zdropped && (flag&KSW_EZ_EXTZ_ONLY) && ez->mqe + end_bonus > (int)ez->max) {
			ez->reach_end = 1;
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->mqe_t, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (ez->max_t >= 0 && ez->max_q >= 0) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		}
		kfree(km, mem2); kfree(km, off);
	}
}

// CUDA线程块大小
#define BLOCK_SIZE 256

__device__ inline int8_t max_int8(int8_t a, int8_t b) {
    return a > b ? a : b;
}

/**
 * 在设备端应用Z-drop检测
 * 用于提前终止比对，如果分数下降太多
 */
__device__ inline int ksw_apply_zdrop_device(
    int32_t max_score, int *max_t, int *max_q,
    int32_t H, int r, int t, int zdrop, int8_t e, int *zdropped)
{
    if (H > max_score) {
        // 更新最大分数和位置
        *max_q = r - t;
        *max_t = t;
        return H;
    } else if (t >= *max_t && r - t >= *max_q) {
        // 计算距离最大分数位置的距离
        int tl = t - *max_t;
        int ql = (r - t) - *max_q;
        int l = tl > ql ? tl - ql : ql - tl;
        
        // 检查是否超过Z-drop阈值
        if (zdrop >= 0 && max_score - H > zdrop + l * e) {
            *zdropped = 1;
            return max_score;
        }
    }
    return max_score;
}

__global__ void ksw_dp_kernel(
    const uint8_t* query,
    const uint8_t* target,
    int qlen, int tlen,
    const int8_t* mat, int8_t m,
    int8_t q, int8_t e,
    int8_t q2, int8_t e2,
    int r,
    int st, int en,
    int8_t* u, int8_t* v,
    int8_t* x, int8_t* y,
    int8_t* x2, int8_t* y2,
    int8_t* s,
    uint8_t* p,
    int flag,
    int with_cigar,
    int n_col)
{
    // 计算全局线程ID，对应对角线上的t坐标
    int t = blockIdx.x * blockDim.x + threadIdx.x + st;
    
    // 边界检查
    if (t > en) return;
    
    // 计算j坐标（query上的位置）
    int j = r - t;
    if (j < 0 || j >= qlen || t >= tlen) return;
    
    // ===== 步骤1：计算匹配/错配分数 =====
    int8_t score;
    if (flag & KSW_EZ_GENERIC_SC) {
        // 通用评分矩阵
        score = mat[target[t] * m + query[qlen - 1 - r + t]];
    } else {
        // 简单的匹配/错配评分
        int8_t sc_mch = mat[0];  // 匹配分数
        int8_t sc_mis = mat[1];  // 错配分数
        int8_t sc_N = (mat[m*m-1] == 0) ? -e2 : mat[m*m-1];  // N的分数
        
        uint8_t tbase = target[t];
        uint8_t qbase = query[qlen - 1 - r + t];
        
        if (tbase == m - 1 || qbase == m - 1) {
            score = sc_N;  // 通配符
        } else {
            score = (tbase == qbase) ? sc_mch : sc_mis;
        }
    }
    s[t] = score;
    
    // ===== 步骤2：加载前一对角线的状态 =====
    // 这些是从r-1对角线来的值
    int8_t xt1, vt1, ut, x2t1;
    
    if (t > 0 && r > 0) {
        xt1 = x[t - 1];   // x[r-1][t-1]
        vt1 = v[t - 1];   // v[r-1][t-1]
        x2t1 = x2[t - 1]; // x2[r-1][t-1]
    } else {
        xt1 = -q - e;
        x2t1 = -q2 - e2;
        vt1 = -q - e;
    }
    
    if (r > 0) {
        ut = u[t];  // u[r-1][t]
    } else {
        ut = -q - e;
    }
    
    // ===== 步骤3：根据Suzuki公式计算z =====
    // z = max{S(i,j), x[r-1][t-1] + v[r-1][t-1], y[r-1][t] + u[r-1][t],
    //         x2[r-1][t-1] + v[r-1][t-1], y2[r-1][t] + u[r-1][t]}
    
    int8_t a = xt1 + vt1;   // 第一套间隙：从对角线来
    int8_t b = y[t] + ut;   // 第一套间隙：从上方来（deletion）
    int8_t a2 = x2t1 + vt1; // 第二套间隙：从对角线来
    int8_t b2 = y2[t] + ut; // 第二套间隙：从上方来
    
    int8_t z = score;
    z = max_int8(z, a);
    z = max_int8(z, b);
    z = max_int8(z, a2);
    z = max_int8(z, b2);
    
    // 限制最大分数（避免溢出）
    int8_t sc_mch = mat[0];
    z = z < sc_mch ? z : sc_mch;
    
    // ===== 步骤4：更新u和v =====
    // u[r][t] = z - v[r-1][t-1]
    // v[r][t] = z - u[r-1][t]
    int8_t u_new = z - vt1;
    int8_t v_new = z - ut;
    
    // ===== 步骤5：计算新的间隙状态 =====
    int8_t tmp;
    
    // 第一套间隙
    tmp = z - q;
    a = a - tmp;  // a = x[r-1][t-1] + v[r-1][t-1] - z + q
    b = b - tmp;  // b = y[r-1][t] + u[r-1][t] - z + q
    
    // 第二套间隙
    tmp = z - q2;
    a2 = a2 - tmp;
    b2 = b2 - tmp;
    
    // 应用max{0, ...} - qe
    int8_t qe = q + e;
    int8_t qe2 = q2 + e2;
    
    int8_t x_new = (a > 0 ? a : 0) - qe;
    int8_t y_new = (b > 0 ? b : 0) - qe;
    int8_t x2_new = (a2 > 0 ? a2 : 0) - qe2;
    int8_t y2_new = (b2 > 0 ? b2 : 0) - qe2;
    
    // ===== 步骤6：存储结果 =====
    u[t] = u_new;
    v[t] = v_new;
    x[t] = x_new;
    y[t] = y_new;
    x2[t] = x2_new;
    y2[t] = y2_new;
    
    // ===== 步骤7：如果需要，记录回溯信息 =====
    if (with_cigar && p != NULL) {
        uint8_t path = 0;
        
        // 确定哪个状态产生了最大值
        // bit 0-2: 哪个状态得到max - 0:H, 1:E, 2:F, 3:E2, 4:F2
        if (flag & KSW_EZ_RIGHT) {
            // 右对齐间隙
            if (z <= a) path = 1;
            else if (z <= b) path = 2;
            else if (z <= a2) path = 3;
            else if (z <= b2) path = 4;
            else path = 0;
            
            // 记录连续性
            if (a > 0) path |= 0x08;
            if (b > 0) path |= 0x10;
            if (a2 > 0) path |= 0x20;
            if (b2 > 0) path |= 0x40;
        } else {
            // 左对齐间隙
            if (a > z) path = 1;
            else if (b > z) path = 2;
            else if (a2 > z) path = 3;
            else if (b2 > z) path = 4;
            else path = 0;
            
            if (a > 0) path |= 0x08;
            if (b > 0) path |= 0x10;
            if (a2 > 0) path |= 0x20;
            if (b2 > 0) path |= 0x40;
        }
        
        // 存储到回溯矩阵
        // p[(size_t)r * n_col + t - st] = path;
        int idx = r * n_col + (t - st);
        p[idx] = path;
    }
}

/**
 * 计算H分数的kernel（用于精确max和Z-drop检测）
 */
__global__ void ksw_compute_H_kernel(
    int r,
    int st0, int en0,
    const int8_t* v,
    const int8_t* u,
    int32_t* H,
    int8_t qe)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x + st0;
    
    if (t > en0) return;
    
    if (r > 0 && t > 0) {
        // H[t] += v[t]
        H[t] = H[t] + (int32_t)v[t];
    } else if (r == 0 && t == 0) {
        H[0] = v[0] - qe;
    }
}

/**
 * 查找H数组中的最大值（用于Z-drop）
 */
__global__ void ksw_find_max_H_kernel(
    const int32_t* H,
    int st0, int en0,
    int32_t* max_H,
    int* max_t)
{
    // 使用共享内存进行规约
    __shared__ int32_t sdata_H[BLOCK_SIZE];
    __shared__ int sdata_t[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int t = blockIdx.x * blockDim.x + threadIdx.x + st0;
    
    // 加载数据到共享内存
    if (t <= en0) {
        sdata_H[tid] = H[t];
        sdata_t[tid] = t;
    } else {
        sdata_H[tid] = KSW_NEG_INF;
        sdata_t[tid] = -1;
    }
    __syncthreads();
    
    // 规约找最大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            if (sdata_H[tid] < sdata_H[tid + s]) {
                sdata_H[tid] = sdata_H[tid + s];
                sdata_t[tid] = sdata_t[tid + s];
            }
        }
        __syncthreads();
    }
    
    // 第一个线程写回结果
    if (tid == 0) {
        atomicMax(max_H, sdata_H[0]);
        if (sdata_H[0] == *max_H) {
            *max_t = sdata_t[0];
        }
    }
}

/**
 * CUDA版本的双仿射间隙惩罚序列比对
 * 
 * 这是主入口函数，将整个比对过程转移到GPU上执行
 */
int ksw_extd2_cuda(
    int qlen, const uint8_t* query,
    int tlen, const uint8_t* target,
    int8_t m, const int8_t* mat,
    int8_t q, int8_t e,
    int8_t q2, int8_t e2,
    int w, int zdrop,
    int end_bonus, int flag,
    ksw_extz_t *ez)
{
    // ===== 初始化 =====
    ksw_reset_extz(ez);
    
    if (m <= 1 || qlen <= 0 || tlen <= 0) return 0;
    
    // 确保q+e不大于q2+e2
    if (q2 + e2 < q + e) {
        int8_t tmp = q; q = q2; q2 = tmp;
        tmp = e; e = e2; e2 = tmp;
    }
    
    int with_cigar = !(flag & KSW_EZ_SCORE_ONLY);
    int approx_max = !!(flag & KSW_EZ_APPROX_MAX);
    
    // 计算带宽
    if (w < 0) w = (tlen > qlen) ? tlen : qlen;
    
    // 计算long_thres和long_diff（用于边界条件）
    int long_thres = (e != e2) ? (q2 - q) / (e - e2) - 1 : 0;
    if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
        ++long_thres;
    int long_diff = long_thres * (e - e2) - (q2 - q) - e2;
    
    int8_t qe = q + e;
    int8_t qe2 = q2 + e2;
    
    // 计算列数（用于存储）
    int n_col = ((tlen < w + 1 ? tlen : w + 1) + 15) / 16 + 1;
    n_col = n_col * 16; // 对齐
    
    // ===== GPU内存分配 =====
    uint8_t *d_query, *d_target;
    int8_t *d_mat;
    int8_t *d_u, *d_v, *d_x, *d_y, *d_x2, *d_y2, *d_s;
    uint8_t *d_p = NULL;
    int32_t *d_H = NULL;
    
    cudaMalloc(&d_query, qlen);
    cudaMalloc(&d_target, tlen);
    cudaMalloc(&d_mat, m * m);
    
    // 为每个对角线分配状态数组
    cudaMalloc(&d_u, tlen);
    cudaMalloc(&d_v, tlen);
    cudaMalloc(&d_x, tlen);
    cudaMalloc(&d_y, tlen);
    cudaMalloc(&d_x2, tlen);
    cudaMalloc(&d_y2, tlen);
    cudaMalloc(&d_s, tlen);
    
    if (!approx_max) {
        cudaMalloc(&d_H, tlen * sizeof(int32_t));
    }
    
    if (with_cigar) {
        // 回溯矩阵：(qlen + tlen - 1) 个对角线，每个最多n_col个元素
        size_t p_size = (size_t)(qlen + tlen - 1) * n_col;
        cudaMalloc(&d_p, p_size);
        cudaMemset(d_p, 0, p_size);
    }
    
    // ===== 拷贝数据到GPU =====
    cudaMemcpy(d_query, query, qlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, tlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, mat, m * m, cudaMemcpyHostToDevice);
    
    // 初始化状态数组
    cudaMemset(d_u, -q - e, tlen);
    cudaMemset(d_v, -q - e, tlen);
    cudaMemset(d_x, -q - e, tlen);
    cudaMemset(d_y, -q - e, tlen);
    cudaMemset(d_x2, -q2 - e2, tlen);
    cudaMemset(d_y2, -q2 - e2, tlen);
    
    if (!approx_max) {
        int32_t neg_inf = KSW_NEG_INF;
        int32_t* h_H_init = (int32_t*)malloc(tlen * sizeof(int32_t));
        for (int i = 0; i < tlen; i++) h_H_init[i] = neg_inf;
        cudaMemcpy(d_H, h_H_init, tlen * sizeof(int32_t), cudaMemcpyHostToDevice);
        free(h_H_init);
    }
    
    // ===== 主循环：遍历所有对角线 =====
    int last_st = -1, last_en = -1;
    int max_score = 0, max_t = -1, max_q = -1;
    int zdropped = 0;
    
    for (int r = 0; r < qlen + tlen - 1 && !zdropped; ++r) {
        // 计算该对角线的边界
        int st = 0, en = tlen - 1;
        
        // 考虑序列长度限制
        if (st < r - qlen + 1) st = r - qlen + 1;
        if (en > r) en = r;
        
        // 考虑带宽限制
        if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1;
        if (en > (r + w) >> 1) en = (r + w) >> 1;
        
        if (st > en) {
            ez->zdropped = 1;
            break;
        }
        
        int st0 = st, en0 = en;
        
        // 设置边界条件
        if (st == 0 && r > 0) {
            int8_t v_init = (r < long_thres) ? -e : 
                           (r == long_thres) ? long_diff : -e2;
            cudaMemcpy(d_v, &v_init, 1, cudaMemcpyHostToDevice);
        }
        
        if (en >= r && r < tlen) {
            int8_t y_init = -q - e;
            int8_t y2_init = -q2 - e2;
            int8_t u_init = (r == 0) ? -q - e :
                           (r < long_thres) ? -e :
                           (r == long_thres) ? long_diff : -e2;
            
            cudaMemcpy(d_y + r, &y_init, 1, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y2 + r, &y2_init, 1, cudaMemcpyHostToDevice);
            cudaMemcpy(d_u + r, &u_init, 1, cudaMemcpyHostToDevice);
        }
        
        // ===== 启动kernel计算这个对角线 =====
        int n_elements = en0 - st0 + 1;
        int n_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        ksw_dp_kernel<<<n_blocks, BLOCK_SIZE>>>(
            d_query, d_target,
            qlen, tlen,
            d_mat, m,
            q, e, q2, e2,
            r, st0, en0,
            d_u, d_v, d_x, d_y, d_x2, d_y2,
            d_s, d_p,
            flag, with_cigar, n_col
        );
        
        cudaDeviceSynchronize();
        
        // ===== 计算H分数和检测Z-drop =====
        if (!approx_max && r > 0) {
            // 更新H数组
            ksw_compute_H_kernel<<<n_blocks, BLOCK_SIZE>>>(
                r, st0, en0, d_v, d_u, d_H, qe
            );
            cudaDeviceSynchronize();
            
            // 找最大值（简化版本，实际应该用更高效的规约）
            int32_t* h_H = (int32_t*)malloc(tlen * sizeof(int32_t));
            cudaMemcpy(h_H, d_H, tlen * sizeof(int32_t), cudaMemcpyDeviceToHost);
            
            int32_t curr_max = h_H[st0];
            int curr_max_t = st0;
            for (int t = st0; t <= en0; t++) {
                if (h_H[t] > curr_max) {
                    curr_max = h_H[t];
                    curr_max_t = t;
                }
            }
            
            // 应用Z-drop
            if (curr_max > max_score) {
                max_score = curr_max;
                max_t = curr_max_t;
                max_q = r - curr_max_t;
            } else if (curr_max_t >= max_t && (r - curr_max_t) >= max_q) {
                int tl = curr_max_t - max_t;
                int ql = (r - curr_max_t) - max_q;
                int l = tl > ql ? tl - ql : ql - tl;
                if (zdrop >= 0 && max_score - curr_max > zdrop + l * e2) {
                    zdropped = 1;
                    ez->zdropped = 1;
                }
            }
            
            // 检查是否到达末端
            if (en0 == tlen - 1 && h_H[en0] > ez->mte) {
                ez->mte = h_H[en0];
                ez->mte_q = r - en0;
            }
            
            if (r - st0 == qlen - 1 && h_H[st0] > ez->mqe) {
                ez->mqe = h_H[st0];
                ez->mqe_t = st0;
            }
            
            if (r == qlen + tlen - 2 && en0 == tlen - 1) {
                ez->score = h_H[tlen - 1];
            }
            
            free(h_H);
        }
        
        last_st = st;
        last_en = en;
    }
    
    // 保存最终结果
    ez->max = max_score;
    ez->max_t = max_t;
    ez->max_q = max_q;
    
    // ===== 清理GPU内存 =====
    cudaFree(d_query);
    cudaFree(d_target);
    cudaFree(d_mat);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_x2);
    cudaFree(d_y2);
    cudaFree(d_s);
    if (d_H) cudaFree(d_H);
    if (d_p) cudaFree(d_p);
    
    return ez->score;
}