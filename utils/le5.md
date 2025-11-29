## 1. Ideia central de Programação Dinâmica

Definição prática:

> DP é um jeito de resolver problemas quebrando em subproblemas menores, garantindo que cada subproblema é resolvido **uma única vez** e reutilizado depois. 

Ela aparece quando:

1. A solução ótima pode ser obtida a partir de soluções ótimas de subproblemas (subestrutura ótima). ([GeeksforGeeks][1])
2. Existem subproblemas que se repetem (subproblemas sobrepostos).

Do ponto de vista de implementação, quase sempre você começa com uma **recursão** e depois:

* salva os resultados dos subproblemas → **memoization (top-down)** ([cp-algorithms.com][2])
* ou preenche uma tabela iterativamente → **tabulation (bottom-up)** ([cp-algorithms.com][2])

Na aula, isso aparece como:

* **estado**: atributos necessários e suficientes para descrever o subproblema
* **transições**: como um estado depende de outros
* **casos base**: estados triviais cuja resposta é direta 

Complexidade típica de uma DP bem modelada:

> `complexidade ≈ (# de estados) × (# de transições por estado)` 

---

## 2. Template genérico (recursivo + memo)

A aula já tinha algo muito parecido: 

```cpp
// dp[state] inicializado com valor "não calculado" (ex: -1)
vector<long long> dp;

long long solve(int state) {
    long long &memo = dp[state];
    if (memo != -1) return memo;             // 1. já calculei?

    if (/* state é caso base */)             // 2. caso base?
        return memo = /* resposta base */;

    long long ans = /* valor neutro (0, -INF, +INF etc.) */;

    for (auto nxt : /* transições a partir de state */) {
        long long cand = /* combinação de solve(nxt) */;
        ans = /* combina ans e cand: max, min, soma... */;
    }

    return memo = ans;
}
```

Versão bottom-up é só ordenar os estados e preencher `dp[state]` na ordem em que as dependências já estejam calculadas.

---

## 3. Checklist para modelar uma DP

Ao ler um problema, tente seguir esse passo a passo:

1. **Que pergunta estou respondendo?**
   Ex.: “qual o maior valor que cabe na mochila?”, “quantas maneiras de chegar na posição X?”.

2. **Consigo descrever o problema com menos informação?**
   Normalmente um prefixo/sufixo/índice + algum parâmetro (capacidade restante, soma atual, máscara de usados etc.)

3. **Defina o estado**

   * “Qual é o menor conjunto de variáveis que, se eu souber, torna o resto do problema independente do passado?”
     Exemplos da aula:
   * Fibonacci: estado = `n` 
   * Knapsack 0/1: estado = `(i, W)` → primeiro `i` itens, capacidade restante `W` 
   * LCS: estado = `(i, j)` → LCS de `s[i:]` e `t[j:]` 

4. **Descubra as transições**
   “A partir desse estado, que escolhas posso fazer?”

   * pegar / não pegar item
   * descer para filho na árvore
   * avançar índice em string
   * etc.

5. **Defina casos base**

   * índice passou do fim
   * capacidade = 0
   * nó folha
   * limite de máscara alcançado etc.

6. **Calcule a resposta e a ordem**

   * `top-down`: deixa a recursão cuidar disso
   * `bottom-up`: encontre uma ordenação topológica dos estados (por índice, por soma, por tamanho de intervalo, profundidade da árvore, máscara crescente etc.)

7. **Calcule complexidade**

   * conte quantos estados existem
   * quantas transições há por estado
   * verifique se cabe no limite do problema (`~10^7–10^8` operações em geral).

---

## 4. Como reconhecer problemas de DP em provas/contests

Indicadores fortes:

* pede **máximo/mínimo** (lucro, pontuação, número de itens etc.)
* pede **quantidade de maneiras** de fazer algo (“em quantos jeitos dá pra…”)
* fala muito de **prefixos, sufixos, subsequências, subarrays**
* pede caminho em grade/grafo, sem pesos negativos, e não é só BFS/DFS simples
* existem restrições que inviabilizam brute force direto, mas ainda são “pequenas” em algum parâmetro (n ≤ 2000, soma ≤ 10^5, máscara com até 2^20 etc.) ([GeeksforGeeks][1])

Padrões comuns:

| Sinal no enunciado                                         | Tipo de DP provável         |
| ---------------------------------------------------------- | --------------------------- |
| “Maior valor, menor custo, melhor pontuação”               | DP de otimização (max/min)  |
| “Quantas maneiras de …”, “número de sequências”            | DP de contagem (com módulo) |
| “Subsequência, subsequência crescente, subsequência comum” | LCS, LIS, subsequências     |
| “Mochila, capacidade, limite de peso”                      | Knapsack / subset sum       |
| “Intervalos, dividir em segmentos, palíndromos”            | Interval DP                 |
| “Dígitos, números ≤ N com propriedade X”                   | Digit DP                    |
| “árvore, subárvores, raiz, filhos”                         | Tree DP                     |
| “subconjunto de vértices/itens pequeno (< 20)”             | Bitmask DP                  |

---

## 5. Padrões clássicos com código

Vou fazer uma minilib `namespace dplib` com funções reutilizáveis (C++17+).

### 5.1 Fibonacci: DP 1D básica

Da aula: versão ingênua é exponencial (árvore de recursão enorme) e com memo cai para O(n). 

```cpp
namespace dplib {

    const long long INF64 = (1LL<<60);

    // Top-down (memoization)
    long long fib_memo(int n, vector<long long> &memo) {
        if (memo[n] != -1) return memo[n];
        if (n < 2) return memo[n] = 1; // F(0) = F(1) = 1 (como na aula)
        return memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo);
    }

    long long fib_memo(int n) {
        vector<long long> memo(n+1, -1);
        return fib_memo(n, memo);
    }

    // Bottom-up (tabulation)
    long long fib_iter(int n) {
        if (n < 2) return 1;
        long long f0 = 1, f1 = 1;
        for (int i = 2; i <= n; ++i) {
            long long f2 = f0 + f1;
            f0 = f1;
            f1 = f2;
        }
        return f1;
    }

}
```

Exemplo de uso:

```cpp
int n; 
cin >> n;
cout << dplib::fib_iter(n) << "\n";
```

---

### 5.2 DP 1D de contagem: “subir escadas”

**Problema:** dado `n`, número de maneiras de subir uma escada de `n` degraus, podendo subir 1 ou 2 de cada vez.

Modelagem:

* estado: `dp[i]` = nº de maneiras de chegar no degrau `i`
* transição: `dp[i] = dp[i-1] + dp[i-2]`
* base: `dp[0] = 1`, `dp[1] = 1`

```cpp
namespace dplib {

    long long stairs_ways(int n) {
        if (n == 0) return 1;
        vector<long long> dp(n+1, 0);
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; ++i) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }

}
```

---

### 5.3 Knapsack 0/1 (clássico)

Da aula: estado `(W, i)` → capacidade restante W e itens a partir do índice i; transições: pegar ou não pegar o item i, complexidade O(W·n). 
Esse é o clássico 0/1 knapsack também descrito em cp-algorithms e Wikipedia. ([cp-algorithms.com][3])

#### Versão top-down

```cpp
namespace dplib {

    long long knapsack01_rec(int i, int W,
                             const vector<int> &w,
                             const vector<int> &v,
                             vector<vector<long long>> &dp) {
        if (i == (int)w.size() || W == 0) return 0;

        long long &memo = dp[i][W];
        if (memo != -1) return memo;

        // opção 1: não pegar item i
        long long ans = knapsack01_rec(i+1, W, w, v, dp);

        // opção 2: pegar item i (se couber)
        if (w[i] <= W) {
            ans = max(ans, (long long)v[i] +
                            knapsack01_rec(i+1, W - w[i], w, v, dp));
        }
        return memo = ans;
    }

    long long knapsack01(const vector<int> &w,
                          const vector<int> &v,
                          int W) {
        int n = (int)w.size();
        vector<vector<long long>> dp(n, vector<long long>(W+1, -1));
        return knapsack01_rec(0, W, w, v, dp);
    }

}
```

#### Versão bottom-up com otimização de espaço

```cpp
namespace dplib {

    long long knapsack01_iter(const vector<int> &w,
                              const vector<int> &v,
                              int W) {
        int n = (int)w.size();
        vector<long long> dp(W+1, 0); // dp[cap] = melhor valor com essa cap

        for (int i = 0; i < n; ++i) {
            // iteramos W decrescendo para não reaproveitar item i mais de uma vez
            for (int cap = W; cap >= w[i]; --cap) {
                dp[cap] = max(dp[cap],
                              dp[cap - w[i]] + v[i]);
            }
        }
        return dp[W];
    }

}
```
```cpp
/*8<
  @Title:

    Binary Knapsack (bottom up)

  @Description:

    Given the points each element have, and it
    repespective cost, computes the maximum points
we can get if we can ignore/choose an element, in
such way that the sum of costs don't exceed the
maximum cost allowed.

  @Time:

    $O(N*W)$

  @Space:

    $O(N*W)$

  @Warning:

    The vectors $VS$ and $WS$ starts at one,
    so it need an empty value at index 0.
>8*/

const int MAXN(1'000), MAXCOST(1'000 * 20);
ll dp[MAXN + 1][MAXCOST + 1];
bool ps[MAXN + 1][MAXCOST + 1];
pair<ll, vi> knapsack(const vll &points, const vi &costs, int maxCost) {
    int n = len(points) - 1;  // ELEMENTS START AT INDEX 1 !

    for (int m = 0; m <= maxCost; m++) {
        dp[0][m] = 0;
    }

    for (int i = 1; i <= n; i++) {
        dp[i][0] = dp[i - 1][0] + (costs[i] == 0) * points[i];
        ps[i][0] = costs[i] == 0;
    }

    for (int i = 1; i <= n; i++) {
        for (int m = 1; m <= maxCost; m++) {
            dp[i][m] = dp[i - 1][m], ps[i][m] = 0;
            int w = costs[i];
            ll v = points[i];

            if (w <= m and dp[i - 1][m - w] + v > dp[i][m]) {
                dp[i][m] = dp[i - 1][m - w] + v, ps[i][m] = 1;
            }
        }
    }

    vi is;
    for (int i = n, m = maxCost; i >= 1; --i) {
        if (ps[i][m]) {
            is.emplace_back(i);
            m -= costs[i];
        }
    }

    return {dp[n][maxCost], is};
}
```
---

### 5.4 LCS – Longest Common Subsequence

Na aula: estado `(i, j)` representando LCS de `s[i:]` e `t[j:]` com transições que permitem pular caractere em uma ou outra string, e quando `s[i] == t[j]` adicionar 1. 
Esse é exatamente o padrão LCS clássico. ([Wikipedia][4])

#### Versão bottom-up

```cpp
namespace dplib {

    int lcs(const string &s, const string &t) {
        int n = (int)s.size();
        int m = (int)t.size();
        vector<vector<int>> dp(n+1, vector<int>(m+1, 0));

        for (int i = n-1; i >= 0; --i) {
            for (int j = m-1; j >= 0; --j) {
                dp[i][j] = max(dp[i+1][j], dp[i][j+1]);
                if (s[i] == t[j])
                    dp[i][j] = max(dp[i][j], 1 + dp[i+1][j+1]);
            }
        }
        return dp[0][0];
    }

}
```

---

### 5.5 LIS – Longest Increasing Subsequence

Na aula aparecem duas abordagens: O(n²) com `dp[i] = tamanho da LIS terminando em i` e O(n log n) usando vetor auxiliar + busca binária. 
O mesmo padrão é apresentado em CP-Algorithms. ([cp-algorithms.com][5])

#### O(n²) – DP simples

```cpp
namespace dplib {

    int lis_n2(const vector<int> &a) {
        int n = (int)a.size();
        if (n == 0) return 0;

        vector<int> dp(n, 1); // LIS terminando em i

        int ans = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (a[j] < a[i]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }

}
```

#### O(n log n) – vetor auxiliar + lower_bound

Na aula, o vetor `aux` não guarda a LIS em si, mas o menor possível último valor de uma subsequência de tamanho i. 

```cpp
namespace dplib {

    int lis_nlogn(const vector<int> &a) {
        vector<int> aux; // aux[k] = menor valor final de uma LIS de tamanho k+1

        for (int x : a) {
            auto it = lower_bound(aux.begin(), aux.end(), x);
            if (it == aux.end()) {
                aux.push_back(x);        // cria LIS maior
            } else {
                *it = x;                 // melhora final de uma LIS desse tamanho
            }
        }
        return (int)aux.size();
    }

}
```

---

### 5.6 Tree DP básico (ex: tamanho da subárvore)

A aula mostra um exemplo mais avançado: **tree matching** máximo usando dois estados por nó (`dp[0][i]`, `dp[1][i]`). 
Antes disso, um padrão base de tree DP é aprender a calcular algo recursivamente nas subárvores.

**Problema:** dado uma árvore enraizada em 0, calcular o tamanho de cada subárvore.

* estado: `subsz[u]` = número de nós na subárvore de `u`
* transição: `subsz[u] = 1 + soma(subsz[v])` para todos filhos `v`

```cpp
namespace dplib {

    void dfs_subtree(int u, int p,
                     const vector<vector<int>> &adj,
                     vector<int> &subsz) {
        subsz[u] = 1;
        for (int v : adj[u]) {
            if (v == p) continue;
            dfs_subtree(v, u, adj, subsz);
            subsz[u] += subsz[v];
        }
    }

    vector<int> subtree_sizes(const vector<vector<int>> &adj) {
        int n = (int)adj.size();
        vector<int> subsz(n, 0);
        dfs_subtree(0, -1, adj, subsz);
        return subsz;
    }

}
```

Esse padrão (DFS que retorna um valor combinado dos filhos) é o esqueleto de praticamente qualquer tree DP.

Se quiser adaptar para o **tree matching** da aula, a ideia é ter `dp[0][u]` e `dp[1][u]` e fazer duas passadas sobre os filhos, como mostrado nos slides. 

---

### 5.7 DP com bitmask (TSP simplificado)

Padrão muito comum para N ≤ 20.

**Problema:** dado um grafo completo com `n` cidades e custo `dist[i][j]`, achar custo mínimo de um ciclo Hamiltoniano (TSP).

* estado: `dp[mask][u]` = custo mínimo de estar no vértice `u` tendo visitado o conjunto de vértices `mask`.
* transição: de `dp[mask][u]`, você tenta ir para cada `v` não visitado:
  `dp[mask | (1<<v)][v] = min(dp[mask | (1<<v)][v], dp[mask][u] + dist[u][v])`.

```cpp
namespace dplib {

    long long tsp_bitmask(const vector<vector<long long>> &dist) {
        int n = (int)dist.size();
        int FULL = (1 << n);
        const long long INF = (1LL<<60);

        vector<vector<long long>> dp(FULL, vector<long long>(n, INF));
        dp[1][0] = 0; // começa na cidade 0, mask=1<<0

        for (int mask = 0; mask < FULL; ++mask) {
            for (int u = 0; u < n; ++u) {
                if (!(mask & (1 << u))) continue;
                if (dp[mask][u] == INF) continue;
                for (int v = 0; v < n; ++v) {
                    if (mask & (1 << v)) continue;
                    int nmask = mask | (1 << v);
                    dp[nmask][v] = min(dp[nmask][v],
                                       dp[mask][u] + dist[u][v]);
                }
            }
        }

        long long ans = INF;
        for (int u = 0; u < n; ++u) {
            ans = min(ans, dp[FULL-1][u] + dist[u][0]);
        }
        return ans;
    }

}
```

---

### 5.8 Interval DP (modelo genérico)

Padrão: `dp[l][r]` resolve algo no intervalo `[l, r]`, e você tenta todas as quebras `k` em `[l, r)`.

Esqueleto:

```cpp
namespace dplib {

    // Exemplo de molde; a lógica dentro depende do problema
    void interval_dp_example(const vector<int> &a) {
        int n = (int)a.size();
        const long long INF = (1LL<<60);
        vector<vector<long long>> dp(n, vector<long long>(n, 0));

        // comprimento crescente
        for (int len = 2; len <= n; ++len) {
            for (int l = 0; l + len - 1 < n; ++l) {
                int r = l + len - 1;
                long long best = INF;
                for (int k = l; k < r; ++k) {
                    // exemplo estilo matrix-chain:
                    // best = min(best, dp[l][k] + dp[k+1][r] + custo(l, r));
                }
                dp[l][r] = best;
            }
        }
    }

}
```

Esse padrão aparece em: Matrix Chain Multiplication, triangulação de polígono, partição ótima de palíndromos etc. ([GeeksforGeeks][1])

---

### 5.9 Digit DP (esqueleto)

Padrão para contar números ≤ N que satisfazem alguma condição de dígitos.

Estados clássicos:

* `pos`: posição do dígito que estou processando
* `tight`: se ainda estou preso ao prefixo de N (0/1)
* `leading_zero`: se ainda não coloquei dígito não-zero
* mais algum parâmetro (soma dos dígitos, resto mod m etc.)

```cpp
namespace dplib {

    long long digit_dp_rec(int pos, bool tight, bool leading,
                           int soma,
                           const string &s,
                           vector<vector<vector<vector<long long>>>> &dp) {
        if (pos == (int)s.size()) {
            // checa condição final sobre soma
            return (soma % 3 == 0) ? 1 : 0; // exemplo
        }

        long long &memo = dp[pos][tight][leading][soma];
        if (memo != -1) return memo;

        int limit = tight ? (s[pos] - '0') : 9;
        long long ans = 0;

        for (int dig = 0; dig <= limit; ++dig) {
            bool ntight = tight && (dig == limit);
            bool nleading = leading && (dig == 0);
            int nsoma = soma;
            if (!nleading) nsoma = (soma + dig) % 3; // exemplo
            ans += digit_dp_rec(pos+1, ntight, nleading, nsoma, s, dp);
        }

        return memo = ans;
    }

    long long digit_dp_example(long long N) {
        string s = to_string(N);
        // dp[pos][tight][leading][soma]
        vector dp(s.size(),
                  vector(2, vector(2, vector<long long>(3, -1))));
        return digit_dp_rec(0, true, true, 0, s, dp);
    }

}
```
```cpp
/*8<
  @Title:

    Edit Distance

  @Time:

    $O(N*M)$

  @Space:

    $O(N*M)$
>8*/

#include "../Contest/template.cpp"

ll edit_distance(const string &a, const string &b) {
    int n = a.size();
    int m = b.size();
    vll2d dp(n + 1, vi(m + 1, 0));

    const ll ADD = 1, DEL = 1, CHG = 1;
    for (int i = 0; i <= n; ++i) {
        dp[i][0] = i * DEL;
    }
    for (int i = 1; i <= m; ++i) {
        dp[0][i] = ADD * i;
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int add = dp[i][j - 1] + ADD;
            int del = dp[i - 1][j] + DEL;
            int chg = dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]) * CHG;
            dp[i][j] = min({add, del, chg});
        }
    }

    return dp[n][m];
}
```
```cpp
/*8<
  @Title:

    Knapsack

  @Description:

    Finds the maximum score you can achieve,
    given that you have $N$ items, each item has
    a $cost$, a $point$ and a $quantity$, you can
    spent at most $maxcost$ and buy each item the
    maximum quantity it has.

  @Time:

    $O(n \cdot maxcost \cdot \log{maxqtd})$

  @Memory:

    $O(maxcost)$.

>8*/

ll knapsack(const vi &weight, const vll &value, const vi &qtd, int maxCost) {
    vi costs;
    vll values;
    for (int i = 0; i < len(weight); i++) {
        ll q = qtd[i];
        for (ll x = 1; x <= q; q -= x, x <<= 1) {
            costs.eb(x * weight[i]);
            values.eb(x * value[i]);
        }
        if (q) {
            costs.eb(q * weight[i]);
            values.eb(q * value[i]);
        }
    }

    vll dp(maxCost + 1);
    for (int i = 0; i < len(values); i++) {
        for (int j = maxCost; j > 0; j--) {
            if (j >= costs[i]) dp[j] = max(dp[j], values[i] + dp[j - costs[i]]);
        }
    }
    return dp[maxCost];
}
```
```cpp
/*8<
  @Title:

    Longest Increasing Subsequence


  @Description:

    Find the pair $(sz, psx)$ where $sz$ is the
    size of the longest subsequence and $psx$ is
    a vector where $psx_i$ tells the size of the
    longest increase subsequence that ends at
    position $i$. $get_idx$ just tells which
indices could be in the longest increasing
subsequence.

    If you want the "Longest Non Decreasing Subsequence"
    you can just change the lower\_bound to an upper\_bound

  @Time:

    $O(n\log{n})$
>8*/

#include "../Contest/template.cpp"

template <typename T>
pair<int, vi> lis(const vector<T> &xs, int n) {
    vector<T> dp(n + 1, numeric_limits<T>::max());
    dp[0] = numeric_limits<T>::min();

    int sz = 0;
    vi psx(n);

    rep(i, 0, n) {
        int pos = lower_bound(all(dp), xs[i]) - dp.begin();

        sz = max(sz, pos);

        dp[pos] = xs[i];

        psx[i] = pos;
    }

    return {sz, psx};
}

template <typename T>
vi get_idx(vector<T> xs) {
    int n = xs.size();

    auto [sz1, psx1] = lis(xs, n);

    transform(rall(xs), xs.begin(), [](T x) { return -x; });

    auto [sz2, psx2] = lis(xs, n);

    vi ans;
    rep(i, 0, n) {
        int l = psx1[i];
        int r = psx2[n - i - 1];
        if (l + r - 1 == sz1) ans.eb(i);
    }

    return ans;
}
```
```cpp
/*
@Description:  Just a regular lis, but very quick to code
 * */

multiset<int> s;
for (int i = 0; i < n; i++) {
    auto it = s.upper_bound(a[i]);
    if (it != s.end()) s.erase(it);
    s.insert(a[i]);
}
lis = len(s);
```
```cpp
/*8<
  @Title:

    Longest Increasing Subsequence


  @Description:

    Find the pair $(sz, psx)$ where $sz$ is the
    size of the longest subsequence and $psx$ is
    a vector where $psx_i$ tells the size of the
    longest increase subsequence that ends at
    position $i$. $get_idx$ just tells which
indices could be in the longest increasing
subsequence.

    If you want the "Longest Non Decreasing Subsequence"
    you can just change the lower\_bound to an upper\_bound

  @Time:

    $O(n\log{n})$
>8*/

#include "../Contest/template.cpp"

template <typename T>
pair<int, vi> lis(const vector<T> &xs, int n) {
    vector<T> dp(n + 1, numeric_limits<T>::max());
    dp[0] = numeric_limits<T>::min();

    int sz = 0;
    vi psx(n);

    rep(i, 0, n) {
        int pos = lower_bound(all(dp), xs[i]) - dp.begin();

        sz = max(sz, pos);

        dp[pos] = xs[i];

        psx[i] = pos;
    }

    return {sz, psx};
}

template <typename T>
vi get_idx(vector<T> xs) {
    int n = xs.size();

    auto [sz1, psx1] = lis(xs, n);

    transform(rall(xs), xs.begin(), [](T x) { return -x; });

    auto [sz2, psx2] = lis(xs, n);

    vi ans;
    rep(i, 0, n) {
        int l = psx1[i];
        int r = psx2[n - i - 1];
        if (l + r - 1 == sz1) ans.eb(i);
    }

    return ans;
}
```
```cpp
/*8<
  @Title:

    Monery sum

  @Description:

    Find every possible sum using the given values
    only once.
>8*/
set<int> money_sum(const vi &xs) {
    using vc = vector<char>;
    using vvc = vector<vc>;
    int _m = accumulate(all(xs), 0);
    int _n = xs.size();
    vvc _dp(_n + 1, vc(_m + 1, 0));
    set<int> _ans;
    _dp[0][xs[0]] = 1;
    for (int i = 1; i < _n; ++i) {
        for (int j = 0; j <= _m; ++j) {
            if (j == 0 or _dp[i - 1][j]) {
                _dp[i][j + xs[i]] = 1;
                _dp[i][j] = 1;
            }
        }
    }

    for (int i = 0; i < _n; ++i)
        for (int j = 0; j <= _m; ++j)
            if (_dp[i][j]) _ans.insert(j);
    return _ans;
}
```
```cpp
/*8<
  @Title: Sum of Subsets
  @Description:
    Allows you to find if some mask $X$ is a
    super mask of any of the given masks
  @Usage:
    Call $build$ with the $masks$ then it returns
    a vector of bool $V$ where $V_X$ says if $X$
    is a super mask of any of the initial maks

    You can change it to count how many submasks
    of each mask exsists, by changing the bitwise
    or by a plus sign...
  @Time: $O(LOG \cdot 2^{LOG})$
  @Memory: $O(LOG^2 \cdot 2^{LOG})$
  @Warning:
    Remember to set $LOG$ with the highest
    bit possible
>8*/
const int LOG = 20;
vc build(const vi &masks) {
    vc ret(1 << LOG);
    trav(mi, masks) ret[mi] = 1;
    rep(b, 0, LOG) {
        rep(mask, 0, (1 << LOG)) {
            if (mask & (1 << b)) ret[mask] |= ret[mask ^ (1 << b)];
        }
    }
    return ret;
}
```
```cpp
template <typename T>
T steinerCost(const vector<vector<T>> &adj, const vi ks,
              T inf = numeric_limits<T>::max()) {
    int k = len(ks), n = len(adj);
    vector<vector<T>> dp(n, vector<T>(1 << k, inf));
    vi inks(n);
    trav(ki, ks) inks[ki] = 1;

    trav(ki, ks) {
        rep(j, 0, n) {
            if (count(all(ks), j) == 0) {
                dp[j][1 << ki] = adj[ki][j];
            }
        }
    }
    rep(mask, 2, (1 << k)) {
        rep(i, 0, n) {
            if (inks[i]) continue;
            for (int mask2 = (mask - 1) & mask; mask2 >= 1;
                 mask2 = (mask2 - 1) & mask) {
                int mask3 = mask ^ mask2;
                chmin(dp[i][mask], dp[i][mask2] + dp[i][mask3]);
            }
            rep(j, 0, n) {
                if (inks[j]) continue;
                chmin(dp[j][mask], dp[i][mask] + adj[i][j]);
            }
        }
    }
    T ans = inf;
    rep(i, 0, n) chmin(ans, dp[i][(1 << k) - 1]);
    return ans;
}
```
```cpp
/*8<
  @Title:

    Travelling Salesman Problem

  @Time:

    $O(N^2 \cdot 2^N )$

  @Memory:

    $O(N^2 \cdot 2^N)$
>8*/
vll2d dist;
vll memo;
int tsp(int i, int mask, int N) {
    if (mask == (1 << N) - 1) return dist[i][0];
    if (memo[i][mask] != -1) return memo[i][mask];
    int ans = INT_MAX << 1;
    for (int j = 0; j < N; ++j) {
        if (mask & (1 << j)) continue;
        auto t = tsp(j, mask | (1 << j), N) + dist[i][j];
        ans = min(ans, t);
    }
    return memo[i][mask] = ans;
}
```

---

## 6. Exemplos de problemas modelados passo a passo

### 6.1 Exemplo 1 – Knapsack da aula

Relembrando o enunciado:
capacidade W, `n` itens com peso `w[i]` e valor `v[i]`. Qual o maior valor que cabe na mochila? 

**Modelagem:**

* estado: `dp[i][W]` = maior valor usando itens do índice `i` até o fim, com capacidade restante `W`
* transições:

  * não pega item i → `dp[i+1][W]`
  * pega item i (se couber) → `v[i] + dp[i+1][W - w[i]]`
* resposta: `dp[0][Wtotal]`
* complexidade: n estados × W capacidades × O(1) transições = O(nW)

Código recursivo + memo já está em `knapsack01_rec`/`knapsack01` (seção 5.3).

---

### 6.2 Exemplo 2 – LCS da aula

Enunciado: dado `s` e `t`, achar tamanho da maior subsequência comum. 

**Modelagem:**

* estado: `dp[i][j]` = LCS de `s[i:]` e `t[j:]`
* transições:

  * pula caractere em `s`: `dp[i+1][j]`
  * pula caractere em `t`: `dp[i][j+1]`
  * se `s[i] == t[j]`: considerar `1 + dp[i+1][j+1]`
* caso base: se `i` ou `j` chegou ao fim → 0
* resposta: `dp[0][0]`

Código: `lcs` na seção 5.4.

---

### 6.3 Exemplo 3 – LIS da aula

Enunciado: maior subsequência estritamente crescente em `s`. 

**Modelagem O(n²):**

* estado: `dp[i]` = tamanho da LIS terminando em `i`
* transição: `dp[i] = 1 + max(dp[j])` para todo `j < i` com `s[j] < s[i]`
* resposta: max de `dp[i]`

**Modelagem O(n log n):**

* mantemos vetor `aux` que garante:

  * `aux[k]` é o menor possível último elemento de uma subsequência de tamanho `k+1`
  * para cada elemento `v`:

    * se `v > aux.back()` → `aux.push_back(v)`
    * senão, substitui o primeiro `aux[pos] >= v` por `v`

Código já está nas funções `lis_n2` e `lis_nlogn`.

---

### 6.4 Exemplo 4 – Tree Matching da aula (esboço)

A aula mostra um problema mais avançado: máximo matching em árvore.
Ideia: para cada nó `n`, temos dois estados:

* `dp[0][n]`: melhor matching na subárvore de `n` sem usar nenhuma aresta ligando `n` a um filho
* `dp[1][n]`: melhor matching na subárvore de `n` usando exatamente uma aresta ligando `n` a um filho

Transições (em alto nível):

1. Calcular `dp[0][n]` somando `max(dp[0][v], dp[1][v])` de todos os filhos `v`.
2. Para `dp[1][n]`, escolhemos um filho `u` com quem `n` fará match, adicionando 1 (pela aresta) e ajustando a soma dos outros filhos. 

A aula termina com um algoritmo O(n) que usa DFS (`tmAux`) e preenche `dp[0][n]` e depois `dp[1][n]`. 

Esse exemplo é útil para reconhecer que, em árvores, muitas vezes precisamos de **vários estados por nó**, representando diferentes “modos” daquele nó (usado, não usado, emparelhado, etc).
---

# **1. Fundamentos Essenciais**

## 1.1 O que é um grafo

Como visto nas **páginas 3–10 da Aula 3** :

* **Vértices (nós)** representam entidades.
* **Arestas** representam relações.
* Podem ser **direcionadas** (p.5) ou **não direcionadas**.
* Podem ter **pesos** (p.6).
* Conceitos fundamentais:

  * Caminho (p.7)
  * Ciclo (p.8)
  * Componentes (p.9)
  * Pontes (p.10)

Esses conceitos aparecem constantemente como subproblemas de problemas maiores (ciclos, conectividade, caminhos mínimos etc.).

---

# **2. Representação de Grafos**

Explicado nas **páginas 13–18 da Aula 3** .

### **2.1 Matriz de adjacência**

* Vantagens: acesso O(1).
* Desvantagens: O(n²) de memória.
* Use quando **n ≤ 2000** e o grafo é denso.

### **2.2 Lista de adjacência**

* Padrão em programação competitiva.
* Usa O(n + m) memória.
* Acesso rápido aos vizinhos.

### **2.3 Grid como grafo**

* Vértices = células
* Arestas implícitas
* Na **página 24–25** é mostrado como navegar usando vetores `dx`, `dy`. 

---

# **3. Percursos Fundamentais: DFS e BFS**

## 3.1 DFS — Depth First Search

Explicado em profundidade nas **páginas 29–42**.

* Explora recursivamente.
* Complexidade O(V+E).
* Usos clássicos:

  * detectar ciclos (p.67)
  * descobrir componentes
  * encontrar pontes
  * DP em árvore
  * encontrar ciclos e recuperar caminho

Código típico:

```cpp
void dfs(int u){
    vis[u] = true;
    for (auto v : adj[u])
        if (!vis[v]) dfs(v);
}
```

---

## 3.2 BFS — Breadth First Search

Páginas **44–55** 

* Usa fila.
* Visita níveis.
* É **o** algoritmo para menor distância em grafos **sem pesos**.

```cpp
void bfs(int s){
    queue<int> q;
    dist[s] = 0;
    q.push(s);

    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int v : adj[u])
            if(dist[v]==-1){
                dist[v] = dist[u] + 1;
                q.push(v);
            }
    }
}
```

### **BFS multisource**

Páginas **64–65**: colocar vários nós iniciais na fila com dist=0.

---

# **4. Estruturas de Grafos Importantes**

## 4.1 Árvores

Explicado nas **páginas 26–28**.

* Conectada e sem ciclos.
* Tem N-1 arestas.
* Há exatamente um caminho entre dois vértices.

Aplicações:

* DP em árvore
* LCA (não incluso aqui mas fundamental em contests)
* Diameter (p.69–70)

---

# **5. Identificação de Problemas de Grafos**

### Indicadores fortes:

* “quantos componentes?” → BFS/DFS
* “existe ciclo?” → DFS
* “menor distância?” → BFS ou Dijkstra
* “menor custo entre todos pares?” → Floyd-Warshall
* “tem pesos negativos?” → Bellman-Ford
* “conectividade dinâmica?” → DSU
* “conectar tudo com menor custo?” → MST
* “grafo é bipartido?” → BFS colorindo 0/1
* “há ordem válida?” → Toposort

---

# **6. Caminhos Mínimos (Shortest Paths)** — Aula 4

## 6.1 BFS (sem pesos)

Para grafos não ponderados, é ótimo.
Usado em grids e grafos grandes.

---

## 6.2 Dijkstra — pesos positivos

Explicado nas **páginas 7–9 da Aula 4** 

* Usa heap de mínimo (priority_queue).
* Resolve: menor caminho **de 1 fonte para todos**.
* Requer arestas com peso ≥ 0.

Código base (resumido da p.7):

```cpp
priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
dist[s] = 0;
pq.push({0, s});

while(!pq.empty()){
    auto [d,u] = pq.top(); pq.pop();
    if(d != dist[u]) continue;
    for(auto [v,w] : adj[u]){
        if(dist[v] > dist[u] + w){
            dist[v] = dist[u] + w;
            pq.push({dist[v], v});
        }
    }
}
```

---

## 6.3 Bellman-Ford — pesos negativos

Páginas **14–15**:

* Detecta ciclos negativos.
* Complexidade O(N·M).
* Útil quando existe pelo menos uma aresta negativa.

---

## 6.4 Floyd–Warshall — todos pares

Páginas **10–13**.

* Computa distâncias entre todos os pares.
* Complexidade O(N³).

Equação principal:

```
dist[u][v] = min(dist[u][v], dist[u][k] + dist[k][v])
```

---

# **7. Modelagem de Grafos para Problemas Customizados**

As páginas **19–22** da Aula 4 são cruciais.
O ponto central:

> Um vértice do grafo original nem sempre representa corretamente o subproblema.
> Às vezes precisamos **modelar estados** como nós de um grafo novo.

Exemplo da aula: Flight Discount (p.19–21).

* Cada estado é `(nó, cupons_usados)`.
* O grafo vira uma malha de camadas (p.21).
* Dijkstra funciona quando rodado sobre o grafo expandido.

Esse conceito é essencial para problemas com:

* limite de teletransportes
* estados de saúde
* custo que depende do número de passos
* grafos com portas que abrem/fecham
* DP + grafo (graph DP)

---

# **8. DSU — Disjoint Set Union**

Páginas **24–29 da Aula 4**.

Serve para:

* verificar conectividade dinâmica
* MST com Kruskal
* compressão de componentes

### Operações:

* **find(u)** → retorna representante
* **join(u,v)** → une conjuntos

### Otimizações:

* path compression (p.27)
* union by size/rank (p.28)

DSU otimizado roda em **α(n)** (p.29) — praticamente constante.

Código típico:

```cpp
vector<int> parent, sz;

int find(int x){
    if(parent[x] == x) return x;
    return parent[x] = find(parent[x]);
}

void join(int a, int b){
    a = find(a); b = find(b);
    if(a == b) return;
    if(sz[a] < sz[b]) swap(a,b);
    parent[b] = a;
    sz[a] += sz[b];
}
```

---

# **9. MST — Árvore Geradora Mínima**

Páginas **31–33 da Aula 4**.

## 9.1 Definição

Árvore que conecta todos os vértices com o menor custo total.

## 9.2 Kruskal

Precisa de DSU.

* Ordenar as arestas por peso.
* Tentar inseri-las se não formarem ciclo.

Aplicável quando:

* grafo é esparso
* todas arestas disponíveis estão listadas

---

# **10. Padrões Clássicos de Problemas de Grafos**

## 10.1 Bipartição (Bicoloribilidade)

Páginas **57–59**.

* BFS/DFS colorindo com 2 cores.
* Se conflito → não bipartido.

---

## 10.2 Toposort (Ordenação Topológica)

Páginas **60–62**.

* Grafo deve ser DAG.
* Usa o Algoritmo de Kahn (fila dos nós com indegree 0).

Útil para:

* dependências
* scheduling
* DP em DAG

---

## 10.3 Encontrar Ciclos

Página **67**.

* DFS com verificação de retorno para vértices já visitados que não são o pai.

---

## 10.4 Componentes Conexos / Flood Fill

Páginas **71–73**.

* Uma DFS/BFS para cada componente.

---

## 10.5 Diâmetro de Árvore

Páginas **69–70**.

* Fazer BFS/DFS de qualquer nó → acha A
* Fazer BFS/DFS de A → acha B
* Distância A–B é o diâmetro

---

# **11. Como identificar qual algoritmo usar**

| Clue do problema                    | Algoritmo                              |
| ----------------------------------- | -------------------------------------- |
| “menor distância sem peso”          | BFS                                    |
| “menor distância com peso positivo” | Dijkstra                               |
| “tem peso negativo”                 | Bellman-Ford                           |
| “todos pares”                       | Floyd-Warshall                         |
| “conectar tudo com menor custo”     | MST (Kruskal/Prim)                     |
| “conectividade dinâmica”            | DSU                                    |
| “existe ciclo?”                     | DFS                                    |
| “ordenar por dependências”          | Toposort                               |
| “vários pontos iniciais”            | BFS multisource                        |
| “grid com obstáculos”               | BFS/DFS                                |
| “árvore e DP”                       | DFS para tree DP                       |
| “limite de recursos no caminho”     | modelagem de estados (grafo expandido) |

---

# **12. Exemplos de Soluções**

## **Exemplo 1 — Caminho mínimo com cupom (Aula 4)**

Referência: páginas 17–22.

**Modelagem de estados:**

* estado = (nó atual, cupons usados)
* transitions:

  * andar sem usar cupom
  * andar usando cupom (custo/2)

**Grafo expandido:**
Cada nó vira até K+1 nós, conforme a figura da **página 21**.

Complexidade: O(K · (N+M) log (N·K)).
Rodar Dijkstra no grafo expandido.

---

## **Exemplo 2 — Detectar ciclos**

Página 67.

```cpp
bool dfs(int u, int p){
    vis[u]=true;
    for(int v : adj[u]){
        if(!vis[v]){
            if(dfs(v,u)) return true;
        } else if(v!=p){
            return true;
        }
    }
    return false;
}
```

---

## **Exemplo 3 — MST com Kruskal**

```cpp
sort(edges.begin(), edges.end());
for (auto [w,u,v]: edges){
    if(find(u)!=find(v)){
        join(u,v);
        mst += w;
    }
}
```

---

## **Exemplo 4 — Distâncias em grid com obstáculos**

```cpp
queue<pair<int,int>> q;
dist[x][y]=0; q.push({x,y});
while(!q.empty()){
    auto [i,j]=q.front(); q.pop();
    for(int k=0;k<4;k++){
        int ni=i+dx[k], nj=j+dy[k];
        if(valid(ni,nj) && dist[ni][nj]==-1){
            dist[ni][nj]=dist[i][j]+1;
            q.push({ni,nj});
        }
    }
}
---

# **1. Fundamentos de Grafos**

## 1.1 Definições básicas

Conforme apresentado nas **páginas 3–10 da Aula Grafos 1** :

* Grafo: conjunto de **vértices** e **arestas**.
* Pode ser **direcionado** (p.5; imagem mostra setas indicando direção) e **não-direcionado**.
* Pode ter **pesos** (p.6).
* Conceitos essenciais ilustrados nas imagens:

  * Caminho simples (p.7)
  * Ciclos (p.8)
  * Componentes conexas (p.9)
  * Pontes e articulações (p.10)

Esses elementos são recorrentes em problemas de conectividade, busca e análise estrutural.

---

# **2. Representações de Grafos**

Explicado nas **páginas 13–18 da Aula Grafos 1**. 

## 2.1 Matriz de Adjacência

* tabela `n × n`.
* A imagem da página 14 mostra um exemplo de matriz preenchida.
* Uso: grafos densos (E ≈ n²).

## 2.2 Lista de Adjacência

* Vetor de vetores; indicado nas páginas 15–18.
* Representação padrão em contests: O(n + m).

## 2.3 Grafos implícitos em grid

* Pages 24–25 ilustram uma malha 2D com movimentos possíveis.
* Movimentos definidos via vetores `dx`, `dy`.

---

# **3. Algoritmos de Busca: DFS e BFS**

Aula **Buscas**, páginas 4–30. 

---

## 3.1 DFS — Depth First Search

Explicado visualmente nas páginas **29–42 da Aula Grafos 1**.


Uso:

* detectar ciclos
* explorar componentes
* pré-processamento em árvores
* reconstrução de caminhos

Código base:

```cpp
void dfs(int u){
    vis[u] = true;
    for(int v : adj[u])
        if(!vis[v]) dfs(v);
}
```

---

## 3.2 BFS — Breadth First Search

Página **44–55 da Aula Grafos 1** mostra passo a passo com figuras.


Uso:

* menor distância em grafo sem peso
* checar bipartição
* fluxo em grid
* BFS multisource (p.64–65)

Código:

```cpp
void bfs(int s){
    queue<int> q;
    dist[s] = 0;
    q.push(s);

    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int v : adj[u]){
            if(dist[v] == -1){
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
}
```

---

# **4. Estruturas e Propriedades de Grafos**

## 4.1 Árvores

Páginas **26–28 de Grafos 1**.


Características:

* n vértices, n−1 arestas
* exatamente um caminho simples entre quaisquer dois nós
* aplicações frequentes: DP em árvore, LCA, diameter

## 4.2 Detectar Ciclos

Página **67** mostra o algoritmo de ciclo com DFS.


---

# **5. Caminhos Mínimos (Shortest Paths)**

Aula **Grafos 2**, páginas 7–22.


---

## 5.1 BFS — sem pesos

Caso trivial. Usado inclusive em grids (na Aula Buscas, páginas 24–30).

## 5.2 Dijkstra — pesos positivos

Mostrado na página **7**, com diagrama explicando relaxação em heap.


```cpp
priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
dist[s] = 0;
pq.push({0, s});
while(!pq.empty()){
    auto [d,u] = pq.top(); pq.pop();
    if(d != dist[u]) continue;
    for(auto [v,w] : adj[u]){
        if(dist[v] > dist[u] + w){
            dist[v] = dist[u] + w;
            pq.push({dist[v], v});
        }
    }
}
```

---

## 5.3 Bellman–Ford — pesos negativos

Páginas **14–15** explicam e mostram a fórmula de relaxação repetida.


Útil para detectar ciclos negativos.

---

## 5.4 Floyd–Warshall — all pairs

Páginas **10–13** apresentam a tabela evoluindo com o "k intermediário".


Equação:

```
dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
```

---

# **6. Modelagem de Estados como Grafos**

Páginas **19–22 da Aula Grafos 2** discutem, com diagramas, transformar estados em nós.


Exemplo: Flight Discount

* nó vira `(cidade, cupons_usados)`
* cria-se um grafo em camadas
* usa Dijkstra no grafo expandido

Esse padrão é essencial em problemas com:

* teletransportes
* energia limitada
* número máximo de ações especiais
* DP misturada com grafo

---

# **7. DSU — Disjoint Set Union**

Páginas **24–29** mostram find, union, path compression e union by size.


Operações fundamentais:

* **find(x)** retorna representante
* **join(x,y)** une conjuntos

Aplicações:

* conectividade
* MST (Kruskal)
* checar se você cria ciclo

---

# **8. Minimum Spanning Tree (MST)**

Páginas **31–33**.


## Kruskal

* ordenar arestas
* usar DSU para unir somente arestas que não criam ciclo

---

# **9. Reconhecimento Rápido de Problemas de Grafos**

| Sinal no enunciado                  | Algoritmo adequado    |
| ----------------------------------- | --------------------- |
| “menor caminho sem peso”            | BFS                   |
| “menor caminho com pesos positivos” | Dijkstra              |
| “pesos negativos”                   | Bellman–Ford          |
| “distância entre todos os pares”    | Floyd–Warshall        |
| “conectar tudo com menor custo”     | MST                   |
| “conectividade dinâmica”            | DSU                   |
| “tem ciclo?”                        | DFS                   |
| “dependências”                      | Toposort              |
| “grid com obstáculos”               | BFS                   |
| “limite de estados”                 | modelagem por camadas |

---

# **10. Técnicas Auxiliares de Busca (da Aula Buscas)**



Essas técnicas não são grafos em si, mas são usadas para *simular grafos implícitos* e navegar em grandes espaços de estado.

---

## 10.1 Binary Search

Páginas **4–13** explicam binary search com diagramas (p.5).


Aplicações em grafos:

* encontrar o menor valor para o qual existe caminho (ex.: limite de capacidade)
* binary search na resposta

---

## 10.2 Two Pointers

Páginas **17–30** mostram passo a passo com imagens.


Pode ser usado em:

* detectar janelas válidas em caminhos lineares
* grafos que são linhas (trees path queries)

---

## 10.3 Sweep Line + Prefix Sum

Páginas **33–42** explicam como varrer intervalos e responder queries.


Aplicações:

* problemas de eventos
* colisão de segmentos
* intervalos sobre eixos que representam vértices ou tempos

---

## 10.4 Complete Search (Força Bruta)

Páginas **45–50** tratam de geração de permutações, subsets e combinações.


Em grafos:

* tentar todas as ordens de visita (TSP)
* brute force em grafos muito pequenos

---

## 10.5 Backtracking

Páginas **51–75** apresentam:

* estado
* transições
* fazer e desfazer
* poda
* sudoku completo com imagens (p.72–74)

Em grafos:

* encontrar caminhos específicos
* Hamiltonian path
* colorir grafo (n pequeno)
* encontrar ciclo simples específico

---

# **11. Exemplos Modelados**

---

## **Exemplo 1 — Caminho mínimo com cupom (Grafos 2)**

Páginas **17–22**.


Estado:

* `(u, cupons_usados)`

Transições:

* andar sem cupom
* andar com cupom (metade do preço)

Usa Dijkstra no grafo expandido.

---

## **Exemplo 2 — Detectar ciclo em grafo não direcionado**

Página **67 da Aula Grafos 1**.


Código:

```cpp
bool dfs(int u, int p){
    vis[u] = true;
    for(int v : adj[u]){
        if(!vis[v]){
            if(dfs(v,p)) return true;
        } else if(v != p)
            return true;
    }
    return false;
}
```

---

## **Exemplo 3 — MST com Kruskal**

```cpp
sort(edges.begin(), edges.end());
for(auto &e : edges){
    if(find(e.u) != find(e.v)){
        join(e.u, e.v);
        total += e.w;
    }
}
```

---

## **Exemplo 4 — BFS em grid (Buscas, páginas 24–30)**



```cpp
queue<pair<int,int>> q;
dist[sx][sy] = 0;
q.push({sx,sy});
while(!q.empty()){
    auto [i,j] = q.front(); q.pop();
    for(int k=0;k<4;k++){
        int ni=i+dx[k], nj=j+dy[k];
        if(valid(ni,nj) && dist[ni][nj]==-1){
            dist[ni][nj]=dist[i][j]+1;
            q.push({ni,nj});
        }
    }
}
```

---

## **Exemplo 5 — Toposort (Grafos 1, páginas 60–62)**



```cpp
queue<int> q;
for(int i=0;i<n;i++)
    if(indegree[i]==0) q.push(i);

while(!q.empty()){
    int u=q.front(); q.pop();
    for(int v:adj[u]){
        if(--indegree[v]==0) q.push(v);
    }
}
```

---

# **12. Checklist de Identificação Rápida**

* Tem pesos?

  * todos positivos → Dijkstra
  * negativos → BF
* Não tem pesos → BFS
* Quer conectar tudo → MST
* Quer saber se está conectado → DFS/DSU
* Quer ordem sem ciclos → Toposort
* Estados limitados → modelagem por camadas
* Grid → BFS
* Restrições pequenas → Backtracking

---

# **13. Mini-Biblioteca de Grafos (C++ Essencial)**

```cpp
// BFS
vector<vector<int>> adj;
vector<int> dist(n,-1);
queue<int> q;

// DFS
void dfs(int u){
    vis[u]=1;
    for(int v:adj[u])
        if(!vis[v]) dfs(v);
}

// Dijkstra
vector<vector<pair<int,int>>> adjW;
vector<long long> dist(n,INF);

// DSU
vector<int> parent, sz;

// Kruskal
struct Edge{int u,v; long long w;};
```
