#pragma once
#include <bits/stdc++.h>
using namespace std;

/*
  Padrões de Binary Search para contest.

  Ideia geral:
    - Precisamos de uma função monótona f(x):
        first_true:  f(x) = false,false,...,true,true,...
        last_true :  f(x) = true,true,...,false,false,...
    - Defina um intervalo "ruim" e outro "bom" (ou vice-versa) e mantenha invariantes.
    - CUIDADO com overflow em mid: use l + (r - l) / 2 (ou checagens de limites).
*/

// ===============================
// 1) Espaço de respostas (inteiro)
// ===============================

// Maior x no intervalo [lo, hi] tal que f(x) == true.
// Se não existir true no intervalo, retorna lo-1 (sentinela).
template <class F>
long long binary_last_true(long long lo, long long hi, F f) {
    // Invariantes: f(lo) pode ser false; f(hi) pode ser true; não assumimos nada.
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2; // viés à direita
        if (f(mid)) lo = mid;   // mid é possível → move lo
        else        hi = mid - 1;
    }
    // Aqui lo == hi; checar se é válido
    return f(lo) ? lo : (lo - 1);
}

// Menor x no intervalo [lo, hi] tal que f(x) == true.
// Se não existir true no intervalo, retorna hi+1 (sentinela).
template <class F>
long long binary_first_true(long long lo, long long hi, F f) {
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2; // viés à esquerda
        if (f(mid)) hi = mid;   // mid funciona → pode ser resposta, aperta hi
        else        lo = mid + 1;
    }
    return f(lo) ? lo : (lo + 1);
}

// ==================================================
// 2) Wrappers para vetor ordenado (lower/upper bound)
// ==================================================

// Retorna o índice do primeiro elemento >= x (ou n se não existir)
template <class T>
int lower_idx(const vector<T>& v, const T& x) {
    return int(lower_bound(v.begin(), v.end(), x) - v.begin());
}

// Retorna o índice do primeiro elemento > x (ou n se não existir)
template <class T>
int upper_idx(const vector<T>& v, const T& x) {
    return int(upper_bound(v.begin(), v.end(), x) - v.begin());
}

// Existe x no vetor ordenado?
template <class T>
bool exists_sorted(const vector<T>& v, const T& x) {
    int i = lower_idx(v, x);
    return i < (int)v.size() && v[i] == x;
}

// ========================================
// 3) Espaço de respostas com double/real
// ========================================

/*
  Busca por ponto em real: encontra o limite mais à direita onde f(x) é true,
  com precisão eps. Supõe que [lo, hi] contém a região útil.
*/
template <class F>
double binary_real_last_true(double lo, double hi, F f, double eps = 1e-9, int iters = 100) {
    // Use número fixo de iterações OU eps
    for (int it = 0; it < iters; ++it) {
        double mid = (lo + hi) / 2.0;
        if (f(mid)) lo = mid; else hi = mid;
        if (hi - lo <= eps) break;
    }
    return lo;
}

/*
  Versão para encontrar o primeiro true (limite à esquerda).
*/
template <class F>
double binary_real_first_true(double lo, double hi, F f, double eps = 1e-9, int iters = 100) {
    for (int it = 0; it < iters; ++it) {
        double mid = (lo + hi) / 2.0;
        if (f(mid)) hi = mid; else lo = mid;
        if (hi - lo <= eps) break;
    }
    return hi;
}

// ====================================
// 4) Utilidades e padrões de sentinela
// ====================================

// Padrão “lo = -1, hi = n” (semântica: lo é ruim, hi é bom)
template <class F>
int first_true_sentinela(int n, F good) {
    int lo = -1, hi = n; // good(lo)=false, good(hi)=true (garantir!)
    while (hi - lo > 1) {
        int mid = lo + (hi - lo) / 2;
        if (good(mid)) hi = mid; else lo = mid;
    }
    return hi; // pode retornar n se não existir good
}

// Padrão “lo = 0, hi = MAX” para resposta inteira máxima válida
// Requer garantir monotonicidade de f.
template <class F>
long long max_answer(long long lo, long long hi, F ok) {
    // assume ok(lo) = true e ok(hi) = false, ou ajuste as bordas antes
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (ok(mid)) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// 1) floor_sqrt(n): maior x tal que x*x <= n
long long floor_sqrt(long long n) {
    auto ok = [&](long long x) {
        if (x < 0) return false;
        // evita overflow: compare x <= n / x
        return x <= 0 ? (x == 0) : (x <= n / x);
    };
    return binary_last_true(0, n, ok);
}

// 2) “Answer Binary Search”: capacidade mínima para transportar tudo em D dias (clássico)
long long min_capacidade(const vector<int>& w, int D) {
    long long lo = *max_element(w.begin(), w.end());   // precisa carregar o maior item
    long long hi = accumulate(w.begin(), w.end(), 0LL); // limite superior trivial

    auto good = [&](long long cap) {
        long long dias = 1, carga = 0;
        for (int x : w) {
            if (carga + x > cap) { dias++; carga = 0; }
            carga += x;
        }
        return dias <= D; // true se essa capacidade funciona
    };
    return binary_first_true(lo, hi, good); // menor cap que funciona
}

// 3) Busca em vetor ordenado
int busca_idx(const vector<int>& a, int x) {
    int i = lower_idx(a, x);
    return (i < (int)a.size() && a[i] == x) ? i : -1;
}

// 4) Ponto de corte em double: achar t máximo tal que f(t) == true
double maior_t_true() {
    auto ok = [&](double t) {
        // Ex.: checar se cabe no orçamento, se velocidade >= alvo, etc.
        // coloque sua condição aqui
        return t*t <= 2.0; // exemplo: t <= sqrt(2)
    };
    return binary_real_last_true(0.0, 10.0, ok, 1e-9);
}
