## Estruturas Básicas e Representação de Grafos

Os grafos podem ser representados de várias formas, duas das mais comuns para competições são:

- **Lista de adjacência:** para cada vértice, guarda-se uma lista dos vértices vizinhos. É eficiente em espaço para grafos esparsos.
- **Matriz de adjacência:** uma matriz onde a posição (i, j) indica se existe uma aresta entre os vértices i e j. Boa para grafos pequenos.

```cpp
// Lista de adjacência básica (não ponderado)
const int MAXN = 1e5+5;
vector<int> adj[MAXN];  // adj[u] armazena todos os vértices conectados a u

// Matriz de adjacência (para grafos pequenos)
int mat[MAXN][MAXN];  // mat[u][v] = 1 se existe aresta entre u e v

// Lista de arestas
vector<pair<int,int>> edges;

// Adicionar aresta não direcionada entre u e v
void add_edge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u); // Remover para grafo direcionado
    edges.push_back({u, v});
}
```

***

## Busca em Grafos (BFS e DFS)

Essas buscas visitam vértices conectados a partir de um vértice inicial.

- **BFS (Busca em Largura):** visita os vértices em "camadas" de distância crescente do vértice inicial. Útil para encontrar caminho mais curto em grafos não ponderados.

```cpp
// BFS retorna vetor de distâncias a partir de start
vector<int> bfs(int start, int n) {
    vector<bool> visited(n+1, false);
    vector<int> dist(n+1, -1);  // Distância para cada vértice (-1 = não alcançado)
    queue<int> q;
    q.push(start);
    visited[start] = true;
    dist[start] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                dist[v] = dist[u]+1;
                q.push(v);
            }
        }
    }
    return dist;
}
```

- **DFS (Busca em Profundidade):** explora um caminho o máximo possível antes de voltar. Usada para componentes conexos, ciclos, ordenação topológica etc.

```cpp
void dfs(int u, vector<bool> &visited) {
    visited[u] = true;
    for (int v : adj[u]) {
        if (!visited[v]) dfs(v, visited);
    }
}
```

***

## Componentes Conexos

Um componente conexo é um conjunto de vértices onde cada par está conectado por algum caminho.

Para encontrar componentes, usa-se DFS marcando os vértices visitados com um índice de componente.

```cpp
vector<int> componente(MAXN, -1);
void dfs_cmp(int u, int cmp) {
    componente[u] = cmp;
    for (int v : adj[u]) {
        if (componente[v] == -1) dfs_cmp(v, cmp);
    }
}
// Na prática, itere por todos os vértices, para aqueles ainda sem componente:
// int ncomp = 0;
// for (int i = 1; i <= n; i++)
//   if (componente[i] == -1) dfs_cmp(i, ncomp++);
```

***

## Bicolorabilidade (Grafos Bipartidos)

Um grafo é bipartido se você pode dividir seus vértices em dois grupos, sem arestas entre vértices do mesmo grupo.

Para checar, usa-se DFS ou BFS e tenta-se colorir alternadamente:

```cpp
vector<int> color(MAXN, -1);
bool is_bipartite_dfs(int u, int c) {
    color[u] = c;
    for (int v : adj[u]) {
        if (color[v] == -1) {
            if (!is_bipartite_dfs(v, 1-c)) return false;
        } else if (color[v] == color[u]) return false; // conflito
    }
    return true;
}
```

***

## Ordenação Topológica

Válida para grafos direcionados acíclicos (DAGs). Permite linearizar os vértices respeitando as direções das arestas.

- **Kahn:** Usa graus de entrada e fila.

```cpp
vector<int> kahn_toposort(int n) {
    vector<int> in_deg(n+1, 0);
    for (int u = 1; u <= n; u++)
        for (int v : adj[u]) in_deg[v]++;
    queue<int> q;
    for (int i = 1; i <= n; i++)
        if (in_deg[i] == 0) q.push(i);
    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (auto v : adj[u]) {
            if (--in_deg[v] == 0) q.push(v);
        }
    }
    return order; // Se ordem.size() != n, o grafo tem ciclo
}
```

- **DFS:** preenche ordem em pós-ordem (invertendo ao final).

```cpp
vector<int> topo;
vector<bool> visited_t(MAXN, false);

void dfs_topo(int u) {
    visited_t[u] = true;
    for (int v: adj[u]) if (!visited_t[v]) dfs_topo(v);
    topo.push_back(u);
}
```

***

```cpp

## Detecção de Ciclos (em grafo direcionado)

Para detectar ciclos, utiliza-se DFS mantendo uma pilha:

- Se visitar um vértice em processo (na pilha), ciclo existe.

```cpp
vector<bool> in_stack(MAXN, false);

bool dfs_ciclo(int u, vector<bool>& visited) {
    visited[u] = true;
    in_stack[u] = true;
    for (int v : adj[u]) {
        if (!visited[v] && dfs_ciclo(v, visited)) return true;
        else if (in_stack[v]) return true;
    }
    in_stack[u] = false;
    return false;
}
```
---
```cpp
#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <bits/stdc++.h>
using namespace std;

/*
    Graph Library - baseada em:
      - definição de grafos (vértices, arestas, peso, direção)
      - representações: lista de adj, matriz, lista de arestas
      - conceitos: caminho, ciclo, componente conexa, grau, DAG, bipartido, árvores
      - DFS (busca em profundidade), grid, etc.

    Filosofia:
      - Focada em competição e ensino.
      - 0-based por padrão (vértices em [0, n-1]).
      - Suporta dirigido/não-dirigido, com/sem peso.
      - Funções utilitárias estáticas para evitar boilerplate no código do aluno.
*/

namespace gr {

// =========================================================
// Tipos básicos
// =========================================================

using Vertex = int;
using Weight = long long;
const Weight INF = (Weight)4e18;

struct Edge {
    Vertex u, v;
    Weight w;
    Edge() {}
    Edge(Vertex _u, Vertex _v, Weight _w = 1) : u(_u), v(_v), w(_w) {}
};

// =========================================================
// Enum e helpers
// =========================================================

enum GraphType {
    UNDIRECTED = 0,
    DIRECTED   = 1
};

// =========================================================
// Classe Graph (lista de adjacência)
// =========================================================

struct Graph {
    int n;                     // número de vértices
    GraphType type;
    bool weighted;
    vector<vector<Edge>> adj;  // adj[u] = lista de arestas saindo de u

    Graph(int _n = 0, GraphType _type = UNDIRECTED, bool _weighted = false)
        : n(_n), type(_type), weighted(_weighted), adj(_n) {}

    void reset(int _n, GraphType _type = UNDIRECTED, bool _weighted = false) {
        n = _n;
        type = _type;
        weighted = _weighted;
        adj.assign(n, {});
    }

    // adiciona aresta u -> v (e v -> u se não-dirigido)
    void add_edge(int u, int v, Weight w = 1) {
        if (!weighted) w = 1;
        adj[u].push_back(Edge(u, v, w));
        if (type == UNDIRECTED) {
            adj[v].push_back(Edge(v, u, w));
        }
    }

    // lista de arestas
    vector<Edge> edges() const {
        vector<Edge> es;
        es.reserve(n * 2);
        for (int u = 0; u < n; ++u) {
            for (auto &e : adj[u]) {
                if (type == DIRECTED || e.u <= e.v) {
                    es.push_back(e);
                }
            }
        }
        return es;
    }

    // matriz de adjacência booleana (existe aresta?)
    vector<vector<bool>> adjacency_matrix_bool() const {
        vector<vector<bool>> mat(n, vector<bool>(n, false));
        for (int u = 0; u < n; ++u)
            for (auto &e : adj[u])
                mat[u][e.v] = true;
        return mat;
    }

    // matriz de adjacência com peso (INF = sem aresta)
    vector<vector<Weight>> adjacency_matrix_weight() const {
        vector<vector<Weight>> mat(n, vector<Weight>(n, INF));
        for (int i = 0; i < n; ++i) mat[i][i] = 0;
        for (int u = 0; u < n; ++u)
            for (auto &e : adj[u])
                mat[u][e.v] = min(mat[u][e.v], e.w);
        return mat;
    }

    // grau de saída de um vértice
    int out_degree(int u) const {
        return (int)adj[u].size();
    }

    // grau de entrada (O(m))
    vector<int> indegrees() const {
        vector<int> indeg(n, 0);
        for (int u = 0; u < n; ++u)
            for (auto &e : adj[u])
                indeg[e.v]++;
        return indeg;
    }

    // tipo simples? (sem laço, sem arestas múltiplas)
    bool is_simple() const {
        // checa laço e múltipla
        for (int u = 0; u < n; ++u) {
            vector<int> seen(n, 0);
            for (auto &e : adj[u]) {
                if (e.v == u) return false; // laço
                if (seen[e.v]) return false; // múltipla
                seen[e.v] = 1;
            }
        }
        return true;
    }

    // grafo completo? (apenas verifica estrutura, ignora pesos)
    bool is_complete() const {
        if (type == UNDIRECTED) {
            // cada vértice deve ser adjacente a todos os outros
            auto mat = adjacency_matrix_bool();
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    if (u == v) continue;
                    if (!mat[u][v]) return false;
                }
            }
            return true;
        } else {
            // versão dirigida: todas as arestas u->v para u!=v
            auto mat = adjacency_matrix_bool();
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v)
                    if (u != v && !mat[u][v]) return false;
            return true;
        }
    }

    // =====================================================
    // BFS (caminho mínimo em arestas unitárias, camadas)
    // =====================================================

    vector<int> bfs(int s) const {
        vector<int> dist(n, -1);
        queue<int> q;
        dist[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &e : adj[u]) {
                int v = e.v;
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        return dist;
    }

    // BFS + caminho reconstruído (arestas de peso 1)
    vector<int> shortest_path_unweighted(int s, int t) const {
        vector<int> dist(n, -1), parent(n, -1);
        queue<int> q;
        dist[s] = 0;
        q.push(s);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == t) break;
            for (auto &e : adj[u]) {
                int v = e.v;
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }

        if (dist[t] == -1) return {}; // sem caminho

        vector<int> path;
        for (int v = t; v != -1; v = parent[v])
            path.push_back(v);
        reverse(path.begin(), path.end());
        return path;
    }

    // =====================================================
    // DFS (recursiva e iterativa) + componentes + ciclos
    // =====================================================

    void dfs_recursive_util(int u, vector<int> &vis) const {
        vis[u] = 1;
        for (auto &e : adj[u]) {
            int v = e.v;
            if (!vis[v]) dfs_recursive_util(v, vis);
        }
    }

    // marca vértices alcançáveis a partir de s
    vector<int> dfs_recursive(int s) const {
        vector<int> vis(n, 0);
        dfs_recursive_util(s, vis);
        return vis;
    }

    // iterativa usando pilha (equivalente à recursão)
    vector<int> dfs_iterative(int s) const {
        vector<int> vis(n, 0);
        stack<int> st;
        st.push(s);
        while (!st.empty()) {
            int u = st.top(); st.pop();
            if (vis[u]) continue;
            vis[u] = 1;
            for (auto &e : adj[u]) {
                int v = e.v;
                if (!vis[v]) st.push(v);
            }
        }
        return vis;
    }

    // componentes conexas (para grafo não-dirigido)
    vector<vector<int>> connected_components() const {
        vector<vector<int>> comps;
        vector<int> vis(n, 0);
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                vector<int> comp;
                stack<int> st;
                st.push(i);
                vis[i] = 1;
                while (!st.empty()) {
                    int u = st.top(); st.pop();
                    comp.push_back(u);
                    for (auto &e : adj[u]) {
                        int v = e.v;
                        if (!vis[v]) {
                            vis[v] = 1;
                            st.push(v);
                        }
                    }
                }
                comps.push_back(comp);
            }
        }
        return comps;
    }

    // detecta ciclo em grafo não-dirigido
    bool has_cycle_undirected() const {
        vector<int> vis(n, 0);
        function<bool(int,int)> dfs = [&](int u, int p) {
            vis[u] = 1;
            for (auto &e : adj[u]) {
                int v = e.v;
                if (!vis[v]) {
                    if (dfs(v, u)) return true;
                } else if (v != p) {
                    return true;
                }
            }
            return false;
        };
        for (int i = 0; i < n; ++i)
            if (!vis[i] && dfs(i, -1)) return true;
        return false;
    }

    // detecta ciclo em grafo dirigido (DFS com 3 cores)
    bool has_cycle_directed() const {
        vector<int> color(n, 0); // 0=branco,1=cinza,2=preto
        function<bool(int)> dfs = [&](int u) {
            color[u] = 1;
            for (auto &e : adj[u]) {
                int v = e.v;
                if (color[v] == 1) return true; // back edge
                if (color[v] == 0 && dfs(v)) return true;
            }
            color[u] = 2;
            return false;
        };
        for (int i = 0; i < n; ++i)
            if (color[i] == 0 && dfs(i)) return true;
        return false;
    }

    // =====================================================
    // Bipartido
    // =====================================================

    bool is_bipartite(vector<int> *color_out = nullptr) const {
        vector<int> color(n, -1);
        queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (color[i] == -1) {
                color[i] = 0;
                q.push(i);
                while (!q.empty()) {
                    int u = q.front(); q.pop();
                    for (auto &e : adj[u]) {
                        int v = e.v;
                        if (color[v] == -1) {
                            color[v] = color[u] ^ 1;
                            q.push(v);
                        } else if (color[v] == color[u]) {
                            return false;
                        }
                    }
                }
            }
        }
        if (color_out) *color_out = color;
        return true;
    }

    // =====================================================
    // Topological Sort (DAG)
    // =====================================================

    // Kahn (BFS) - supõe grafo dirigido
    vector<int> topo_sort_kahn() const {
        vector<int> indeg = indegrees();
        queue<int> q;
        for (int i = 0; i < n; ++i)
            if (indeg[i] == 0) q.push(i);

        vector<int> order;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            order.push_back(u);
            for (auto &e : adj[u]) {
                int v = e.v;
                if (--indeg[v] == 0)
                    q.push(v);
            }
        }
        if ((int)order.size() != n) {
            // tem ciclo, não é DAG -> retorna vazio
            return {};
        }
        return order;
    }

    // DFS-based topo sort
    vector<int> topo_sort_dfs() const {
        vector<int> color(n, 0);
        vector<int> order;
        bool has_cycle = false;

        function<void(int)> dfs = [&](int u) {
            color[u] = 1;
            for (auto &e : adj[u]) {
                int v = e.v;
                if (color[v] == 0) dfs(v);
                else if (color[v] == 1) has_cycle = true;
            }
            color[u] = 2;
            order.push_back(u);
        };

        for (int i = 0; i < n; ++i)
            if (color[i] == 0) dfs(i);

        if (has_cycle) return {};
        reverse(order.begin(), order.end());
        return order;
    }

    // =====================================================
    // Dijkstra (pesos não-negativos)
    // =====================================================

    vector<Weight> dijkstra(int s) const {
        vector<Weight> dist(n, INF);
        using P = pair<Weight,int>;
        priority_queue<P, vector<P>, greater<P>> pq;
        dist[s] = 0;
        pq.push({0, s});

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d != dist[u]) continue;
            for (auto &e : adj[u]) {
                int v = e.v;
                Weight nd = d + e.w;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    pq.push({nd, v});
                }
            }
        }
        return dist;
    }

    // =====================================================
    // Bellman-Ford (detecta ciclo negativo)
    // =====================================================

    pair<vector<Weight>, bool> bellman_ford(int s) const {
        vector<Weight> dist(n, INF);
        dist[s] = 0;
        vector<Edge> es = edges();

        for (int i = 0; i < n - 1; ++i) {
            bool changed = false;
            for (auto &e : es) {
                if (dist[e.u] == INF) continue;
                if (dist[e.u] + e.w < dist[e.v]) {
                    dist[e.v] = dist[e.u] + e.w;
                    changed = true;
                }
            }
            if (!changed) break;
        }

        bool neg_cycle = false;
        for (auto &e : es) {
            if (dist[e.u] != INF && dist[e.u] + e.w < dist[e.v]) {
                neg_cycle = true;
                break;
            }
        }

        return {dist, neg_cycle};
    }

    // =====================================================
    // Floyd-Warshall (todas as fontes) - usa matriz
    // =====================================================

    vector<vector<Weight>> floyd_warshall() const {
        auto dist = adjacency_matrix_weight();
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                if (dist[i][k] < INF)
                    for (int j = 0; j < n; ++j)
                        if (dist[k][j] < INF &&
                            dist[i][k] + dist[k][j] < dist[i][j])
                            dist[i][j] = dist[i][k] + dist[k][j];
        return dist;
    }

    // =====================================================
    // Árvores: checagens e utilidades básicas
    // =====================================================

    bool is_tree_undirected() const {
        if (type != UNDIRECTED) return false;
        // árvore: conexo + m = n-1
        auto es = edges();
        if ((int)es.size() != n - 1) return false;
        auto comps = connected_components();
        return (int)comps.size() == 1;
    }

    // assume que é árvore enraizada em root, constrói parent e depth
    void build_parent_depth(int root, vector<int> &parent, vector<int> &depth) const {
        parent.assign(n, -1);
        depth.assign(n, 0);
        stack<int> st;
        st.push(root);
        parent[root] = root;
        while (!st.empty()) {
            int u = st.top(); st.pop();
            for (auto &e : adj[u]) {
                int v = e.v;
                if (v == parent[u]) continue;
                parent[v] = u;
                depth[v] = depth[u] + 1;
                st.push(v);
            }
        }
    }

    // LCA com binary lifting (para árvore enraizada)
    struct LCA {
        int n, LOG;
        vector<vector<int>> up;
        vector<int> depth;

        LCA() {}
        LCA(const Graph &g, int root) {
            build(g, root);
        }

        void build(const Graph &g, int root) {
            n = g.n;
            LOG = 1;
            while ((1 << LOG) <= n) LOG++;
            up.assign(LOG, vector<int>(n, -1));
            depth.assign(n, 0);

            // DFS iterativo
            stack<int> st;
            st.push(root);
            up[0][root] = root;
            depth[root] = 0;

            vector<int> vis(n, 0);
            vis[root] = 1;

            while (!st.empty()) {
                int u = st.top(); st.pop();
                for (auto &e : g.adj[u]) {
                    int v = e.v;
                    if (vis[v]) continue;
                    vis[v] = 1;
                    up[0][v] = u;
                    depth[v] = depth[u] + 1;
                    st.push(v);
                }
            }

            for (int k = 1; k < LOG; ++k)
                for (int v = 0; v < n; ++v)
                    up[k][v] = up[k-1][ up[k-1][v] ];
        }

        int lca(int a, int b) const {
            if (depth[a] < depth[b]) swap(a, b);
            int diff = depth[a] - depth[b];
            for (int k = LOG - 1; k >= 0; --k)
                if (diff & (1 << k))
                    a = up[k][a];
            if (a == b) return a;
            for (int k = LOG - 1; k >= 0; --k) {
                if (up[k][a] != up[k][b]) {
                    a = up[k][a];
                    b = up[k][b];
                }
            }
            return up[0][a];
        }
    };

    // =====================================================
    // Grid Graph helpers (4/8 adj) - útil para slides sobre grid
    // =====================================================

    // Converte grid HxW em grafo (bloqueios opcionais)
    // blocked[y][x] = true -> célula não existe no grafo
    static Graph from_grid_4conn(int H, int W,
                                 const vector<vector<bool>> &blocked) {
        int n = H * W;
        Graph g(n, UNDIRECTED, false);
        auto id = [&](int y, int x) { return y * W + x; };
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (blocked[y][x]) continue;
                int u = id(y, x);
                const int dy[4] = {-1, 1, 0, 0};
                const int dx[4] = {0, 0, -1, 1};
                for (int k = 0; k < 4; ++k) {
                    int ny = y + dy[k];
                    int nx = x + dx[k];
                    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                    if (blocked[ny][nx]) continue;
                    int v = id(ny, nx);
                    if (u < v) g.add_edge(u, v); // evita duplicar
                }
            }
        }
        return g;
    }

    // BFS em grid gerado por from_grid_4conn
    static vector<int> grid_bfs_4conn(int H, int W,
                                      const vector<vector<bool>> &blocked,
                                      int sy, int sx) {
        Graph g = from_grid_4conn(H, W, blocked);
        int s = sy * W + sx;
        return g.bfs(s); // dist em número de passos
    }

}; // struct Graph

} // namespace gr

#endif // GRAPH_HPP
```
---

## 🟩 1. **Busca em Largura (BFS)** – *Caminho mínimo em grafos não ponderados*

**Palavras-chave:**
`menor número de passos`, `distância mínima`, `labirinto`, `nível`, `espalhar`, `onda`, `propagação`, `infectar`, `alcance`

**Exemplos de problemas:**

* **“Labirinto”** → dado um mapa de `#` e `.`, ache o menor número de passos de A até B.
  🔑 *Use BFS em grid 4-direcional.*
* **“Zumbis se espalhando”** → tempo mínimo até infectar todos.
  🔑 *BFS com múltiplas fontes.*
* **“Cavaleiro no tabuleiro de xadrez”** → número mínimo de movimentos de cavalo.
  🔑 *BFS em movimentos de cavalo (8 direções).*

---

## 🟦 2. **Busca em Profundidade (DFS)** – *Componentes, ciclos, e pintura*

**Palavras-chave:**
`componentes`, `regiões`, `ilhas`, `manchas`, `recursivo`, `explorar`, `conectado`, `subgrafo`, `contar grupos`

**Exemplos:**

* **“Ilhas”** → quantas regiões de ‘terra’ existem num mapa binário.
  🔑 *DFS recursivo para marcar células.*
* **“Rede de amigos”** → quantos grupos de amigos distintos há.
  🔑 *DFS sobre grafo não-dirigido.*
* **“Ciclo em grafo”** → verificar se há ciclo em um grafo dado.
  🔑 *DFS com pai (undirected) ou 3 cores (directed).*

---

## 🟨 3. **Bipartição e cores**

**Palavras-chave:**
`duas cores`, `times`, `amigos e inimigos`, `grafo bipartido`, `divisão possível`, `não pode ter ciclo ímpar`

**Exemplo:**

* **“Dois times”** → dado quem não se gosta, é possível dividir em dois grupos sem conflito?
  🔑 *DFS/BFS com coloração 0-1 (bipartido).*

---

## 🟧 4. **Ordenação Topológica (Toposort)**

**Palavras-chave:**
`dependências`, `ordem de execução`, `tarefas`, `pré-requisito`, `DAG`, `sem ciclo`, `precedência`

**Exemplos:**

* **“Ordenar cursos”** → dado que A é pré-requisito de B, encontre uma ordem válida.
  🔑 *Toposort via Kahn (fila) ou DFS.*
* **“Compilação de módulos”** → quais módulos podem ser compilados primeiro?
  🔑 *Grafo dirigido acíclico.*

---

## 🟥 5. **Dijkstra (menor caminho ponderado)**

**Palavras-chave:**
`distância mínima`, `custo`, `estradas`, `pedágio`, `tempo`, `energia`, `peso`, `não negativo`

**Exemplos:**

* **“Rotas entre cidades”** → menor custo de viagem entre A e B.
  🔑 *Dijkstra com priority_queue.*
* **“Entrega rápida”** → minimize o tempo com pesos positivos.
  🔑 *Cada estrada tem custo; evite repetir vértices.*

---

## 🟪 6. **Bellman-Ford / Floyd-Warshall**

**Palavras-chave:**
`custos negativos`, `lucro`, `ciclo negativo`, `arbitragem`, `conversão de moedas`, `todas as distâncias`

**Exemplos:**

* **“Arbitragem de moedas”** → é possível ganhar dinheiro trocando moedas em ciclo?
  🔑 *Bellman-Ford detecta ciclo negativo.*
* **“Rota mínima entre todos os pares”**
  🔑 *Floyd-Warshall para todos-vs-todos.*

---

## 🟫 7. **Árvores**

**Palavras-chave:**
`sem ciclo`, `conexo`, `pai`, `filho`, `ancestral`, `LCA`, `distância na árvore`, `hierarquia`

**Exemplos:**

* **“Empresa Hierárquica”** → dado o organograma, encontre o chefe comum de dois funcionários.
  🔑 *LCA (Lowest Common Ancestor).*
* **“Rede de comunicação”** → atraso máximo entre dois nós.
  🔑 *Diâmetro da árvore via duas DFS.*

---

## 🟩 8. **Grafos em grade (grid)**

**Palavras-chave:**
`labirinto`, `mapa`, `celulas`, `movimentos`, `parede`, `flood fill`, `distância Manhattan`

**Exemplos:**

* **“Labirinto com paredes”** → BFS no grid.
* **“Área de pintura”** → quantas regiões são alcançáveis (DFS).
* **“Fogo e saída”** → BFS com múltiplas fontes (fogo e pessoa).

---

## 🟦 9. **Modelagem de problema**

**Palavras-chave:**
`estado`, `transformações`, `movimentos`, `botões`, `configuração`, `transição`

**Exemplos:**

* **“Quebra-cabeça 8-puzzle”** → vértice = configuração do tabuleiro.
  🔑 *BFS em espaço de estados.*
* **“Botões que alteram bits”** → estados binários como nós.
  🔑 *0-1 BFS ou Dijkstra.*

---

## 🟨 10. **Árvore Geradora Mínima (MST)**

**Palavras-chave:**
`custo mínimo`, `rede elétrica`, `conectar`, `sem ciclo`, `ligar todas`, `estradas`, `construir`

**Exemplos:**

* **“Rede elétrica barata”** → custo mínimo para conectar todas as cidades.
  🔑 *Kruskal (union-find) ou Prim.*

---

## 🟥 11. **Problemas combinados / híbridos**

**Palavras-chave:**
`atalhos`, `teleporte`, `peso 0 ou 1`, `dois níveis de grafo`, `restrições`

**Exemplos:**

* **“Teletransporte e estradas”** → grafo misto, pesos 0 e 1 → *0-1 BFS.*
* **“Matriz com portais”** → grid + grafos → modelar vértices = células.

---

## 🔎 12. **Meta-palavras para reconhecer um problema de grafos**

Em muitos enunciados, o autor *nunca diz “grafo”*, mas usa termos como:

* “Cidades e estradas” → grafo
* “Pessoas e amizades” → grafo não-dirigido
* “Tarefas e dependências” → grafo dirigido
* “Salas conectadas” → grid/grafo
* “Mapa” → grid
* “Rede elétrica / cabos / fios” → MST
* “Fluxo de dados” → grafo dirigido com pesos

---

## 🟩 1. Representação e conceitos básicos

### Tipos de questão

* “Dado o número de vértices e arestas, construa o grafo e responda consultas simples”
  → exemplo: *grau de um vértice, se existe aresta entre u e v, se é completo/simples.*
* “Quantas componentes conexas há?”
* “O grafo é regular / completo / árvore?”

### Dicas

Use lista de adjacência (`vector<vector<int>>`) e BFS/DFS.

---

## 🟦 2. Busca em Largura (BFS)

### Padrões de problema

* **Caminho mínimo em arestas não ponderadas**
  → exemplo: “Qual o menor número de arestas entre A e B?”
* **Labirinto (grid)**
  → cada célula é um vértice; movimentos 4-direcionais.
* **Espalhamento / tempo mínimo para visitar tudo**
  → exemplo: “mínimo de passos para infectar todo o grafo”.

### Problemas típicos

* “Maze” (AtCoder, OBI, UVA 119 - Greedy Gift Givers)
* “Knight Moves” (BFS em grid 8-direcional)
* “Caminho mais curto entre cidades conectadas”

---

## 🟨 3. Busca em Profundidade (DFS)

### Padrões

* **Contar componentes conexas**
* **Detectar ciclos**
* **Verificar se o grafo é bipartido**
* **Topological sort (em DAG)**
* **Flood fill em grid (pintar regiões)**

### Problemas clássicos

* *“Ilhas”* (quantas áreas de terra em um mapa)
* *“Ciclos em grafo dirigido”*
* *“É possível ordenar as tarefas?”*

---

## 🟥 4. Grafos ponderados

### Padrões

* **Dijkstra** → pesos não-negativos
  ex: menor caminho entre cidades com custo de estrada.
* **Bellman-Ford** → permite peso negativo e detecta ciclos.
* **Floyd–Warshall** → distâncias entre todos os pares.

### Problemas típicos

* “Shortest path” / “Caminho mínimo entre dois vértices”
* “Negócios lucrativos” (ciclo negativo = arbitragem)

---

## 🟪 5. Árvores

### Padrões

* Verificar se o grafo é uma árvore.
* Encontrar LCA (lowest common ancestor).
* Calcular diâmetro de uma árvore.
* Percorrer e calcular soma de pesos.

### Problemas típicos

* “Network delay time” (árvore de comunicação)
* “Company hierarchy” (subárvores e ancestrais)
* “Distância entre dois nós em árvore”

---

## 🟧 6. DAG (Grafos Acíclicos Dirigidos)

### Padrões

* **Ordenação topológica**
* **Caminhos mais longos** (em DAGs)
* **Contagem de caminhos possíveis**

### Problemas típicos

* “Ordenação de tarefas”
* “Dependências de pacotes”
* “Longest path in a DAG”

---

## 🟫 7. Modelagem com Grafos

### Padrões

* Problemas de **labirinto** → BFS em grid.
* **Quebra-cabeças** → vértices = estados; arestas = movimentos.
* **Problemas de amigos / redes sociais** → componentes conexas.
* **Teletransporte / portais** → grafos mistos com pesos 0 e 1 → use *0-1 BFS*.

---

## 🧩 8. Avançados (começam a aparecer em nível regional)

* **Union-Find (DSU)** → detectar ciclos, componentes, MST (Kruskal).
* **MST (Árvore Geradora Mínima)** → Kruskal / Prim.
* **Toposort + DP em DAG** → contagem de caminhos, longest path.
* **Bipartite matching** (Hopcroft-Karp) → mais avançado, mas cai.

---

## 🏆 Estratégia de estudo para competições

| Tema                      | Técnica-chave  | Complexidade | Frequência |
| ------------------------- | -------------- | ------------ | ---------- |
| Componentes / DFS         | DFS            | O(V+E)       | Altíssima  |
| Caminho mínimo (sem peso) | BFS            | O(V+E)       | Altíssima  |
| Caminho mínimo (com peso) | Dijkstra       | O(E log V)   | Alta       |
| Ciclos / DAG              | DFS / Toposort | O(V+E)       | Alta       |
| Árvores                   | DFS / LCA      | O(V log V)   | Média      |
| Grid                      | BFS / DFS      | O(H×W)       | Alta       |
| MST                       | Kruskal / Prim | O(E log V)   | Média      |
---
## 1️⃣ Labirinto – Menor caminho em grid (BFS)

**Resumo:**
Dado um grid `N x M` com:

* `'.'` = livre
* `'#'` = parede
* `S` = início
* `T` = destino
  Ache o **menor número de passos** (4 direções). Se não der, imprima `-1`.

**Ideia:**
Modelar cada célula como vértice; arestas entre vizinhos livres. Usar **BFS**.

**Solução (C++):**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    cin >> N >> M;
    vector<string> grid(N);
    for (int i = 0; i < N; i++) cin >> grid[i];

    pair<int,int> S, T;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            if (grid[i][j] == 'S') S = {i,j};
            if (grid[i][j] == 'T') T = {i,j};
        }

    const int INF = 1e9;
    vector<vector<int>> dist(N, vector<int>(M, INF));
    queue<pair<int,int>> q;
    dist[S.first][S.second] = 0;
    q.push(S);

    int dy[4] = {-1,1,0,0};
    int dx[4] = {0,0,-1,1};

    while (!q.empty()) {
        auto [y,x] = q.front(); q.pop();
        for (int k = 0; k < 4; k++) {
            int ny = y + dy[k], nx = x + dx[k];
            if (ny < 0 || ny >= N || nx < 0 || nx >= M) continue;
            if (grid[ny][nx] == '#') continue;
            if (dist[ny][nx] > dist[y][x] + 1) {
                dist[ny][nx] = dist[y][x] + 1;
                q.push({ny,nx});
            }
        }
    }

    int ans = dist[T.first][T.second];
    cout << (ans == INF ? -1 : ans) << "\n";
}
```

Palavras-chave: *labirinto, menor número de passos, mapa, grid*.

---

## 2️⃣ Componentes Conexas – “Grupos de Amigos” (DFS/BFS)

**Resumo:**
Há `N` pessoas e `M` amizades (não-dirigido). Quantos **grupos desconexos** (componentes) existem?

**Ideia:**
Graph não-dirigido, contar componentes com DFS/BFS.

**Solução (C++ com nossa lib):**

```cpp
#include "graph.hpp"
using namespace std;
using namespace gr;

int main() {
    int N, M;
    cin >> N >> M;
    Graph g(N, UNDIRECTED, false);
    while (M--) {
        int a, b;
        cin >> a >> b;
        --a; --b;
        g.add_edge(a,b);
    }
    auto comps = g.connected_components();
    cout << comps.size() << "\n";
}
```

Palavras-chave: *grupos, ilhas, quantos conjuntos, conectados*.

---

## 3️⃣ Verificar se dá pra dividir em dois times (Bipartido)

**Resumo:**
Dado grafo não-dirigido onde arestas representam “não podem ficar no mesmo time”. Verifique se é possível dividir vértices em 2 times sem conflito.

**Ideia:**
Checar se o grafo é **bipartido** (BFS/DFS com 2 cores, conflito = aresta com mesma cor).

**Solução:**

```cpp
#include "graph.hpp"
using namespace std;
using namespace gr;

int main() {
    int N, M;
    cin >> N >> M;
    Graph g(N, UNDIRECTED, false);
    while (M--) {
        int a,b; cin >> a >> b;
        --a; --b;
        g.add_edge(a,b);
    }
    vector<int> color;
    if (g.is_bipartite(&color)) {
        cout << "YES\n";
    } else {
        cout << "NO\n";
    }
}
```

Palavras-chave: *duas cores, dois times, dividir em 2 grupos, bipartido*.

---

## 4️⃣ Ordem das Tarefas – Topological Sort

**Resumo:**
Temos `N` tarefas e `M` dependências `A -> B` (A antes de B).
Pergunta:

* Existe uma ordem válida?
  Se sim, imprima uma.

**Ideia:**
Grafo dirigido. Se é DAG, **toposort** (Kahn); se tiver ciclo, impossível.

**Solução:**

```cpp
#include "graph.hpp"
using namespace std;
using namespace gr;

int main() {
    int N, M;
    cin >> N >> M;
    Graph g(N, DIRECTED, false);
    while (M--) {
        int A, B;
        cin >> A >> B;
        --A; --B;
        g.add_edge(A,B);
    }
    auto ord = g.topo_sort_kahn();
    if (ord.empty()) {
        cout << "IMPOSSIBLE\n";
    } else {
        for (int v : ord) cout << v+1 << " ";
        cout << "\n";
    }
}
```

Palavras-chave: *tarefas, dependências, ordem, precedência, DAG*.

---

## 5️⃣ Caminho Mínimo com Pesos – Rotas entre Cidades (Dijkstra)

**Resumo:**
`N` cidades, `M` estradas com custo positivo. Dado `S` e `T`, calcule o **caminho de menor custo**.

**Ideia:**
Grafo ponderado, pesos não-negativos → **Dijkstra**.

**Solução:**

```cpp
#include "graph.hpp"
using namespace std;
using namespace gr;

int main() {
    int N, M;
    cin >> N >> M;
    int S, T;
    cin >> S >> T;
    --S; --T;
    Graph g(N, UNDIRECTED, true);
    while (M--) {
        int a, b;
        long long w;
        cin >> a >> b >> w;
        --a; --b;
        g.add_edge(a,b,w);
    }
    auto dist = g.dijkstra(S);
    if (dist[T] == INF) cout << -1 << "\n";
    else cout << dist[T] << "\n";
}
```

Palavras-chave: *custo mínimo, distância, pedágio, tempo, rota mais barata*.

---

## 6️⃣ Checar se é Árvore

**Resumo:**
Dado um grafo não-dirigido com `N` vértices e `M` arestas, verifique se ele é uma **árvore**.

**Definição prática de prova:**
É árvore se:

* é conexo
* não tem ciclo
* `M = N - 1`

**Ideia:**
Usar diretamente `is_tree_undirected()` ou checar componentes + M = N-1.

**Solução:**

```cpp
#include "graph.hpp"
using namespace std;
using namespace gr;

int main() {
    int N, M;
    cin >> N >> M;
    Graph g(N, UNDIRECTED, false);
    while (M--) {
        int a,b; cin >> a >> b;
        --a; --b;
        g.add_edge(a,b);
    }
    cout << (g.is_tree_undirected() ? "YES\n" : "NO\n");
}
```

Palavras-chave: *sem ciclos, conexo, N-1 arestas, árvore*.

---

## 7️⃣ LCA – Menor Ancestral Comum em Árvore

**Resumo:**
Árvore enraizada em 1 com `N` nós. Dadas `Q` queries `(u, v)`, responder o **menor ancestral comum**.

**Ideia:**
Preprocessar com LCA (binary lifting), depois responder cada query em `O(log N)`.

**Solução (usando LCA embutido):**

```cpp
#include "graph.hpp"
using namespace std;
using namespace gr;

int main() {
    int N, Q;
    cin >> N >> Q;
    Graph tree(N, UNDIRECTED, false);
    for (int i = 0; i < N-1; i++) {
        int a,b; cin >> a >> b;
        --a; --b;
        tree.add_edge(a,b);
    }
    int root = 0; // vértice 1 na entrada
    Graph::LCA lca(tree, root);

    while (Q--) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        int ans = lca.lca(u, v);
        cout << ans + 1 << "\n";
    }
}
```

Palavras-chave: *ancestral comum, árvore enraizada, hierarquia*.

---

## 8️⃣ MST – Rede Elétrica Mais Barata (Kruskal)

**Resumo:**
Conectar todas as cidades com cabos de custo mínimo. Dado grafo não-dirigido ponderado, ache o custo da **árvore geradora mínima**.

**Ideia:**
Usar **Kruskal** com Union-Find.

**Solução (direta, sem depender da lib anterior):**

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, r;
    DSU(int n): p(n), r(n,0) { iota(p.begin(), p.end(), 0); }
    int findp(int x){ return p[x]==x?x:p[x]=findp(p[x]); }
    bool unite(int a,int b){
        a=findp(a); b=findp(b);
        if(a==b) return false;
        if(r[a]<r[b]) swap(a,b);
        p[b]=a;
        if(r[a]==r[b]) r[a]++;
        return true;
    }
};

struct Edge {
    int u,v;
    long long w;
};

int main() {
    int N,M;
    cin >> N >> M;
    vector<Edge> es(M);
    for(int i=0;i<M;i++){
        cin >> es[i].u >> es[i].v >> es[i].w;
        es[i].u--; es[i].v--;
    }
    sort(es.begin(), es.end(), [](auto &a, auto &b){return a.w<b.w;});
    DSU dsu(N);
    long long cost = 0;
    int used = 0;
    for(auto &e: es){
        if(dsu.unite(e.u,e.v)){
            cost += e.w;
            used++;
        }
    }
    if(used != N-1) cout << "IMPOSSIBLE\n";
    else cout << cost << "\n";
}
```

Palavras-chave: *ligar todas as cidades, custo mínimo, sem ciclo, rede elétrica*.

---

## 9️⃣ 0-1 BFS – Teletransporte + Caminhar

**Resumo:**
Cada aresta tem custo 0 ou 1. Ex: você pode:

* andar para lado com custo 1,
* usar portal com custo 0.
  Achar menor custo de `S` a `T`.

**Ideia:**
Usar **0-1 BFS** (deque), não Dijkstra normal.

**Solução:**

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge { int v, w; };

int main() {
    int N, M;
    cin >> N >> M;
    vector<vector<Edge>> g(N);
    while (M--) {
        int a,b,w;
        cin >> a >> b >> w; // w = 0 ou 1
        --a; --b;
        g[a].push_back({b,w});
        g[b].push_back({a,w});
    }
    int S,T;
    cin >> S >> T;
    --S; --T;

    const int INF = 1e9;
    vector<int> dist(N, INF);
    deque<int> dq;
    dist[S] = 0;
    dq.push_front(S);

    while (!dq.empty()) {
        int u = dq.front(); dq.pop_front();
        for (auto &e : g[u]) {
            if (dist[u] + e.w < dist[e.v]) {
                dist[e.v] = dist[u] + e.w;
                if (e.w == 0) dq.push_front(e.v);
                else dq.push_back(e.v);
            }
        }
    }

    cout << (dist[T] == INF ? -1 : dist[T]) << "\n";
}
```

Palavras-chave: *portais, atalhos, custo 0 e 1, mínimo de cliques / mudanças*.

---

## 🔟 Detectar Ciclo em Digrafo – “Tem dependência circular?”

**Resumo:**
Dado grafo dirigido, dizer se existe algum **ciclo** (dependência circular).

**Ideia:**
DFS com cores (0=branco,1=cinza,2=preto). Se achar aresta para cinza → ciclo.

**Solução (usando nossa lib):**

```cpp
#include "graph.hpp"
using namespace std;
using namespace gr;

int main() {
    int N,M;
    cin >> N >> M;
    Graph g(N, DIRECTED, false);
    while (M--) {
        int a,b; cin >> a >> b;
        --a; --b;
        g.add_edge(a,b);
    }
    cout << (g.has_cycle_directed() ? "YES\n" : "NO\n");
}
```

Palavras-chave: *dependências circulares, não é possível ordenar, ciclo*.

---
