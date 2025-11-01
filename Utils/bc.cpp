// board_toolkit.cpp
// "Códigão" de utilidades para problemas de tabuleiro + exemplos de backtracking
// + solver pronto p/ Avoid Knight Attack (contagem de casas seguras contra cavalo).

#include <bits/stdc++.h>
using namespace std;

// ============================ UTIL / FAST IO ================================
static inline void fast_io() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
}

// Pack de (i,j) em 64 bits (i,j 1-based); N <= 1e9, então 32 bits por coordenada é seguro.
static inline uint64_t pack_pair(uint32_t i, uint32_t j) {
    return (uint64_t(i) << 32) | uint64_t(j);
}

// ============================ POS / BOARD ===================================
struct Pos {
    int r, c; // 0-based internamente na maior parte do toolkit
    Pos() : r(0), c(0) {}
    Pos(int r_, int c_) : r(r_), c(c_) {}
    bool operator==(const Pos& o) const { return r==o.r && c==o.c; }
    bool operator!=(const Pos& o) const { return !(*this==o); }
};

struct RectBoard {
    int R, C;
    RectBoard(int R=0, int C=0) : R(R), C(C) {}
    inline bool in_bounds(int r, int c) const {
        return r>=0 && r<R && c>=0 && c<C;
    }
    inline bool in_bounds(const Pos& p) const { return in_bounds(p.r, p.c); }
};

// ============================ MOVE-SETS (XADREZ) ============================
// Rei: 8 vizinhos
const int KING_DR[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
const int KING_DC[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

// Cavalo: 8 deslocamentos (mesmo padrão do PDF do "Avoid Knight Attack"). :contentReference[oaicite:1]{index=1}
const int KNIGHT_DR[8] = {+2,+1,-1,-2,-2,-1,+1,+2};
const int KNIGHT_DC[8] = {+1,+2,+2,+1,-1,-2,-2,-1};

// Direções deslizantes: torre (4), bispo (4), rainha (8 = torre+bispo)
const int ROOK_DR[4]   = {-1, 0, 1, 0};
const int ROOK_DC[4]   = { 0, 1, 0,-1};
const int BISHOP_DR[4] = {-1,-1, 1, 1};
const int BISHOP_DC[4] = {-1, 1, 1,-1};

// ============================ BACKTRACKING BASE =============================
// Um esqueleto genérico de backtracking no estilo "constrói solução passo-a-passo".
//
// Use assim: defina
//  - next_var(state)             -> qual variável/posição decidir agora
//  - candidates(state, var, out) -> gera as opções viáveis p/ essa variável
//  - apply(state, var, cand)     -> faz a decisão
//  - undo(state, var, cand)      -> desfaz (backtrack)
//  - is_solution(state)          -> checa se solução completa
//
// Exemplo prático nos blocos N-Queens e Knights a seguir.

template <class State, class Var, class Cand>
struct Backtracker {
    function<bool(const State&)> is_solution;
    function<Var(const State&)> next_var;
    function<void(const State&, const Var&, vector<Cand>&)> candidates;
    function<void(State&, const Var&, const Cand&)> apply;
    function<void(State&, const Var&, const Cand&)> undo;
    function<void(const State&)> on_solution = [](const State&){};

    long long nodes = 0;
    long long solutions = 0;
    long long limit_solutions = (long long)4e18; // sem limite

    void dfs(State& st) {
        ++nodes;
        if (is_solution(st)) {
            ++solutions;
            on_solution(st);
            return;
        }
        Var v = next_var(st);
        vector<Cand> opts;
        candidates(st, v, opts);
        for (const Cand& x : opts) {
            apply(st, v, x);
            dfs(st);
            if (solutions >= limit_solutions) return;
            undo(st, v, x);
        }
    }
};

// ============================ N-QUEENS (bitmasks) ===========================
// Colocar N rainhas num tabuleiro NxN sem se atacarem.
// Backtracking com bitmasks -> O(típico) bem baixo.

struct NQueensState {
    int n, row;             // linha corrente a preencher
    vector<int> where;      // where[row] = coluna escolhida
    // colMask: colunas ocupadas; d1Mask: diag principal; d2Mask: diag secundária
    // diag index: d1 = (r - c) + (n-1); d2 = (r + c)
    vector<bool> colUsed, d1Used, d2Used;

    NQueensState(int n): n(n), row(0),
        where(n, -1),
        colUsed(n, false),
        d1Used(2*n-1, false),
        d2Used(2*n-1, false) {}
};

struct NQueensSolver {
    Backtracker<NQueensState,int,int> bt;
    long long total = 0;

    NQueensSolver(int n) {
        bt.is_solution = [](const NQueensState& st){ return st.row == st.n; };
        bt.next_var = [](const NQueensState& st){ return st.row; };
        bt.candidates = [](const NQueensState& st, const int& r, vector<int>& out){
            out.clear();
            for (int c=0;c<st.n;c++){
                int d1 = (r - c) + (st.n - 1);
                int d2 = r + c;
                if (!st.colUsed[c] && !st.d1Used[d1] && !st.d2Used[d2]) out.push_back(c);
            }
        };
        bt.apply = [](NQueensState& st, const int& r, const int& c){
            st.where[r] = c;
            st.colUsed[c] = true;
            st.d1Used[(r - c) + (st.n - 1)] = true;
            st.d2Used[r + c] = true;
            st.row++;
        };
        bt.undo = [](NQueensState& st, const int& r, const int& c){
            st.row--;
            st.where[r] = -1;
            st.colUsed[c] = false;
            st.d1Used[(r - c) + (st.n - 1)] = false;
            st.d2Used[r + c] = false;
        };
        bt.on_solution = [&](const NQueensState&){ total++; };
    }

    long long solve() {
        NQueensState st(bt.is_solution.target_type().hash_code()); // só p/ compilar (não usado)
        st = NQueensState(bt.is_solution.target_type().hash_code()); // arranjo; substituímos já
        // arrumando de fato:
        // (acima foi truque p/ evitar warning: reconstituímos de verdade aqui)
        int n = 8; // valor placeholder, substitua no caller
        return 0;
    }

    // Versão correta, sem firulas:
    static long long count(int n) {
        NQueensState st(n);
        NQueensSolver s(n);
        s.bt.on_solution = [&](const NQueensState&){ s.total++; };
        s.bt.dfs(st);
        return s.total;
    }
};

// ============================ KNIGHT'S TOUR (esqueleto) =====================
// Passeio do Cavalo: visitar todas as casas uma vez.
// Aqui fica só um esqueleto com heurística de Warnsdorff opcional.
// Útil para tabuleiros pequenos/experimentais (8x8, 6x6...).

struct KnightTour {
    int R, C;
    vector<vector<int>> vis;  // -1 = não visitado; senão = ordem da visita
    RectBoard B;
    vector<Pos> path;
    KnightTour(int R, int C): R(R), C(C), vis(R, vector<int>(C, -1)), B(R,C) {}

    // Conta possíveis próximos passos (para Warnsdorff)
    int degree(int r, int c) {
        int d=0;
        for (int k=0;k<8;k++){
            int nr=r+KNIGHT_DR[k], nc=c+KNIGHT_DC[k];
            if (B.in_bounds(nr,nc) && vis[nr][nc]==-1) d++;
        }
        return d;
    }

    bool dfs(int r, int c, int step) {
        vis[r][c] = step;
        path.emplace_back(r,c);
        if (step == R*C-1) return true;

        // gera movimentos
        vector<pair<int,Pos>> nxt; nxt.reserve(8);
        for (int k=0;k<8;k++){
            int nr=r+KNIGHT_DR[k], nc=c+KNIGHT_DC[k];
            if (B.in_bounds(nr,nc) && vis[nr][nc]==-1) {
                nxt.push_back({degree(nr,nc), Pos(nr,nc)});
            }
        }
        // Warnsdorff: ordenar por menor grau
        sort(nxt.begin(), nxt.end(), [](auto& A, auto& B){ return A.first < B.first; });

        for (auto &it : nxt){
            int nr = it.second.r, nc = it.second.c;
            if (dfs(nr,nc,step+1)) return true;
        }
        // backtrack
        vis[r][c] = -1;
        path.pop_back();
        return false;
    }
};

// ============================ K KNIGHTS (exemplo) ===========================
// Exemplo backtracking: colocar K cavalos em NxN sem ataques mútuos.
// (Para N moderado; usa marcação local.)

struct KKnights {
    int n, K, placed;
    vector<vector<bool>> occ, att; // ocupada? / atacada?
    long long ways=0;

    KKnights(int n, int K): n(n), K(K), placed(0), occ(n, vector<bool>(n,false)), att(n, vector<bool>(n,false)) {}

    inline bool safe(int r,int c){
        if (occ[r][c] || att[r][c]) return false;
        return true;
    }
    void mark_knight(int r,int c,bool put){
        occ[r][c] = put;
        for (int k=0;k<8;k++){
            int nr = r+KNIGHT_DR[k], nc = c+KNIGHT_DC[k];
            if (nr>=0 && nr<n && nc>=0 && nc<n) att[nr][nc] = (att[nr][nc] || put);
        }
    }
    void dfs_cell(int idx){
        if (placed == K){ ways++; return; }
        if (idx == n*n) return;
        int r = idx / n, c = idx % n;

        // não colocar aqui
        dfs_cell(idx+1);

        // colocar aqui (se seguro)
        if (safe(r,c)){
            // precisamos marcar de maneira reversível -> salvamos células que mudaremos
            vector<pair<int,int>> changed;
            // set occ e marca ataques (guardando quais eram false e ficaram true)
            occ[r][c] = true;
            for (int k=0;k<8;k++){
                int nr=r+KNIGHT_DR[k], nc=c+KNIGHT_DC[k];
                if (nr>=0 && nr<n && nc>=0 && nc<n){
                    if (!att[nr][nc]){ att[nr][nc]=true; changed.push_back({nr,nc}); }
                }
            }
            placed++;
            dfs_cell(idx+1);
            placed--;
            // desfaz
            occ[r][c] = false;
            for (auto &p: changed) att[p.first][p.second]=false;
        }
    }
};

// ====================== AVOID KNIGHT ATTACK (CONTEÚDO DO PDF) ===============
// Dado N (1<=N<=1e9), M (<=2e5) e M peças (a_k,b_k), quantas casas vazias
// NÃO são atacadas por cavalo a partir de qualquer peça? Solução O(M log M):
//  - Total de casas = N^2 (cabe em uint64).
//  - Marcar todas ocupadas (M) e todas atacadas (máx. 8M) usando hash de pares.
//  - Resposta = N^2 - M - (#ataques distintos válidos).
//
// OBS: Usamos 1-based no input, então verificação de limites é 1..N.

unsigned long long solve_avoid_knight_attack() {
    uint32_t N; int M;
    cin >> N >> M;
    vector<pair<uint32_t,uint32_t>> P(M);
    for (int i=0;i<M;i++){
        uint32_t a,b; cin >> a >> b;
        P[i]={a,b};
    }

    // Conjunto de ocupadas (para não contar posição ocupada como "vazia atacável").
    // Conjunto de atacadas (vazias).
    unordered_set<uint64_t> occ; occ.reserve(M*2);
    unordered_set<uint64_t> atk; atk.reserve(size_t(8)*size_t(M)*2);

    for (auto &q : P) {
        occ.insert(pack_pair(q.first, q.second));
    }
    auto in_bounds_1based = [&](long long r,long long c)->bool{
        return (1LL <= r && r <= N && 1LL <= c && c <= N);
    };

    for (auto &q : P) {
        long long r = q.first, c = q.second;
        for (int k=0;k<8;k++){
            long long nr = r + KNIGHT_DR[k];
            long long nc = c + KNIGHT_DC[k];
            if (in_bounds_1based(nr,nc)) {
                uint64_t key = pack_pair((uint32_t)nr, (uint32_t)nc);
                if (!occ.count(key)) atk.insert(key); // só casas vazias atacadas
            }
        }
    }

    // N^2 cabe em 64 bits sem sinal (1e18 <= 2^64-1).
    unsigned long long total = (unsigned long long)N * (unsigned long long)N;
    unsigned long long used  = (unsigned long long)M + (unsigned long long)atk.size();
    unsigned long long free_safe = total - used;
    return free_safe;
}

// ============================ MAIN: escolha o modo ===========================
// Por padrão: roda o solver do Avoid Knight Attack (do PDF). :contentReference[oaicite:2]{index=2}
// Para testar os outros exemplos, altere os #defines.

#define RUN_AVOID_KNIGHT_ATTACK
// #define RUN_N_QUEENS_COUNT
// #define RUN_K_KNIGHTS_COUNT
// #define RUN_KNIGHT_TOUR_DEMO

int main(){
    fast_io();

#ifdef RUN_AVOID_KNIGHT_ATTACK
    // Entrada: N M, seguido de M linhas (a_k b_k). Saída: número de casas livres e seguras.
    cout << solve_avoid_knight_attack() << "\n";
    return 0;
#endif

#ifdef RUN_N_QUEENS_COUNT
    int n; cin >> n;
    cout << NQueensSolver::count(n) << "\n";
    return 0;
#endif

#ifdef RUN_K_KNIGHTS_COUNT
    int n, K; cin >> n >> K;
    KKnights kk(n,K);
    kk.dfs_cell(0);
    cout << kk.ways << "\n";
    return 0;
#endif

#ifdef RUN_KNIGHT_TOUR_DEMO
    int R,C,sr,sc; // exemplo: 8 8 0 0
    cin >> R >> C >> sr >> sc;
    KnightTour KT(R,C);
    bool ok = KT.dfs(sr,sc,0);
    cout << (ok ? "FOUND\n" : "NO\n");
    if (ok){
        for (auto &p : KT.path) {
            cout << p.r << " " << p.c << "\n";
        }
    }
    return 0;
#endif

    return 0;
}
