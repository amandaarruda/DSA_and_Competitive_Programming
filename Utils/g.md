## SSSP (Caminho Mínimo de uma fonte)
- **Dijkstra:** para grafos sem arestas negativas.

```cpp
#include <vector>
#include <queue>
#include <limits>
using namespace std;
const int INF = numeric_limits<int>::max();
struct Edge { int to, cost; };
vector<int> dijkstra(int n, int src, vector<vector<Edge>> &adj) {
    vector<int> dist(n, INF);
    dist[src] = 0;
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
    pq.emplace(0, src);
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto &e : adj[u]) {
            if (dist[e.to] > d + e.cost) {
                dist[e.to] = d + e.cost;
                pq.emplace(dist[e.to], e.to);
            }
        }
    }
    return dist;
}
```

- **Bellman-Ford:** para grafos com pesos negativos (detecta ciclos negativos).

```cpp
vector<int> bellman_ford(int n, int src, const vector<tuple<int,int,int>> &edges) {
    vector<int> dist(n, INF); dist[src] = 0;
    for (int i = 0; i < n-1; ++i)
        for (auto [u,v,c] : edges)
            if (dist[u] < INF && dist[v] > dist[u] + c)
                dist[v] = dist[u] + c;
    return dist;
}
```

***

## APSP (Todos os pares)
- **Floyd-Warshall:** ideal para grafos menores (até alguns centenas).

```cpp
vector<vector<int>> floyd_warshall(int n, const vector<vector<int>> &mat) {
    vector<vector<int>> dist = mat;
    for(int k = 0; k < n; ++k)
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                if(dist[i][k] < INF && dist[k][j] < INF)
                    dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j]);
    return dist;
}
```

Para grafos grandes, use Dijkstra repetido.

***

## Fluxo Máximo
### 1. Edmonds-Karp (BFS)
```cpp
struct Edge { int to, rev; int cap; };
struct MaxFlow {
    int n; vector<vector<Edge>> adj;
    MaxFlow(int n): n(n), adj(n) {}
    void add_edge(int u, int v, int cap) {
        adj[u].push_back({v, (int)adj[v].size(), cap});
        adj[v].push_back({u, (int)adj[u].size()-1, 0});
    }
    int bfs(int s, int t, vector<int> &parent) {
        fill(parent.begin(), parent.end(), -1);
        queue<pair<int,int>> q; q.push({s, INT_MAX}); parent[s] = -2;
        while (!q.empty()) {
            int u = q.front().first, flow = q.front().second; q.pop();
            for (auto &e : adj[u]) {
                if (parent[e.to] == -1 && e.cap > 0) {
                    parent[e.to] = u;
                    int new_flow = min(flow, e.cap);
                    if (e.to == t) return new_flow;
                    q.push({e.to, new_flow});
                }
            }
        }
        return 0;
    }
    int max_flow(int s, int t) {
        int flow = 0; vector<int> parent(n);
        int new_flow;
        while ((new_flow = bfs(s, t, parent))) {
            flow += new_flow;
            int cur = t;
            while (cur != s) {
                int prev = parent[cur];
                for (auto &e : adj[prev]) if (e.to == cur && e.cap > 0) {
                    e.cap -= new_flow;
                    adj[cur][e.rev].cap += new_flow;
                    break;
                }
                cur = prev;
            }
        }
        return flow;
    }
};
```

### 2. Dinic (BFS + DFS nivelado)
```cpp
struct Edge { int v, rev; long long cap; };
struct Dinic {
    int N; vector<vector<Edge>> adj; vector<int> level, ptr;
    Dinic(int N) : N(N), adj(N) {}
    void add_edge(int u, int v, long long cap) {
        adj[u].push_back({v, (int)adj[v].size(), cap});
        adj[v].push_back({u, (int)adj[u].size()-1, 0});
    }
    bool bfs(int s, int t) {
        level.assign(N, -1); level[s]=0; queue<int> q; q.push(s);
        while(!q.empty()) {
            int u=q.front();q.pop();
            for(auto &e:adj[u])
                if(e.cap>0&&level[e.v]==-1)
                    level[e.v]=level[u]+1,q.push(e.v);
        }
        return level[t]!=-1;
    }
    long long dfs(int u,int t,long long f) {
        if(u==t||f==0) return f;
        for(int &i=ptr[u];i<adj[u].size(); ++i){
            Edge &e=adj[u][i];
            if(level[e.v]!=level[u]+1||e.cap==0)continue;
            long long pushed=dfs(e.v,t,min(f,e.cap));
            if(pushed){ e.cap-=pushed; adj[e.v][e.rev].cap+=pushed; return pushed; }
        } return 0;
    }
    long long max_flow(int s, int t){
        long long flow=0;
        while(bfs(s, t)){
            ptr.assign(N,0);
            while(long long f=dfs(s,t,1e18)) flow+=f;
        } return flow;
    }
};
```

***

**Dica:** Para problemas menores ou onde descrição rápida é importante, Edmonds-Karp funciona. Para grafos grandes, use Dinic para melhor desempenho.

---
## Reconhecimento de Problemas Clássicos: SSSP, APSP e Fluxo Máximo

### 1. **SSSP (Single Source Shortest Path) — Caminho mínimo a partir de uma fonte**
**Quando aparece:**
- O enunciado pede "menor custo/menor caminho" do vértice $$s$$ para todos os outros ou para um destino $$t$$.
- Palavras-chave comuns: "caminho mais curto entre A e B", "custo mínimo para chegar", "menor distância a partir de X".

**Macete:** só existe uma fonte! Geralmente aparece um vértice origem especificado (às vezes "cidade inicial", "start") e pode ser grafo dirigido ou não.

**Como decidir o algoritmo:**
- Se não há arestas negativas: Dijkstra.
- Se podem existir negativas: Bellman-Ford.
- Se todas arestas valem 1: BFS padrão resolve.

**Problema clássico:**
- **"Menor caminho de X para todos"**
- **"Entrega mais rápida partindo de X"**

***

### 2. **APSP (All Pairs Shortest Path) — Caminho mínimo entre todos os pares**
**Quando aparece:**
- O problema pergunta "menor caminho entre todo par de vértices" ou pede matriz de distâncias de todos para todos.
- Palavras-chave: "menor custo entre quaisquer duas cidades", "cálculo de todas as distâncias".

**Macete:** aparecem perguntas como "qual o menor caminho entre qualquer par?" ou pede comparar distâncias de vários pares.

**Como decidir o algoritmo:**
- Se o grafo é pequeno (~400 vértices): **Floyd-Warshall**.
- Se o grafo é grande, use Dijkstra para cada vértice (caso sem negativos).

**Problema clássico:**
- **"Todas as menores distâncias de X para Y ($$\forall $$X,Y)"**
- **"Determinar pares de vértices com menor custo de conexão"**

***

### 3. **Fluxo Máximo**
**Quando aparece:**
- O enunciado envolve **capacidades** (restrições de quantidade entre vértices), e pede "quantidade máxima que pode passar", ou "maximizar X sem violar limites".
- Palavras-chave: "fluxo máximo", "capacidade máxima", "máximo número/chave de rotas sem sobrepor", "roteamento máximo de produtos/pessoas".

**Macete:** geralmente envolve duas regiões ($$source$$ e $$sink$$), ou "enviar de A para B o máximo possível sem exceder capacidades nas conexões".

**Como decidir o algoritmo:**
- Para casos simples e instâncias menores/media: **Edmonds-Karp** (BFS)
- Para grafos grandes ou muitos queries de fluxo: **Dinic**

**Problema clássico:**
- **"Qual o volume máximo do bem de X para Y pelas rotas?"**
- **"Máximo casamento em gráficos bipartidos"**
- **"Problema do emparelhamento máximo"**

***

## **Resumo prático: heurísticas rápidas**
- Tem só uma origem? Pede menores caminhos? → **SSSP**
- Pede menores caminhos entre todos os pares? → **APSP**
- Fala em limitações de fluxo/capacidade e quer maximizar transporte? → **Fluxo Máximo**

---
1. Base de grafo
// base.hpp
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll INFLL = (ll)4e18;

// Grafo ponderado (para Dijkstra, Bellman-Ford, fluxo etc.)
struct Edge {
    int to;
    ll w;
};

2. SSSP (Single-Source Shortest Path)
2.1 BFS em grafo não ponderado

Serve para:

Maze 6x6 (grid + paredes)

Commandos (distância em edges)


763497

Template BFS simples:

vector<int> bfs_unweighted(int n, int s, const vector<vector<int>>& adj) {
    vector<int> dist(n, -1);
    queue<int> q;
    dist[s] = 0;
    q.push(s);
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int u : adj[v]) {
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                q.push(u);
            }
        }
    }
    return dist;
}

2.1.1 BFS em grid + reconstrução de caminho (Maze)

Pro Maze 6×6: cada célula (r,c) vira um índice id = r*6 + c. Você monta adjacência respeitando as paredes, e depois reconstrói o path para imprimir N,E,S,W.

struct GridPath {
    int n,m;
    vector<string> dir; // direção do passo que leva até aqui
    vector<int> parent;

    GridPath(int n, int m): n(n), m(m), dir(n*m, '?'), parent(n*m, -1) {}

    // exemplo: BFS simples em grid sem paredes, só pra ilustrar
    string solve(pair<int,int> start, pair<int,int> goal, 
                 function<bool(int,int,int,int)> can_move) {
        static int dr[4] = {-1,0,1,0};
        static int dc[4] = {0,1,0,-1};
        static char ch[4] = {'N','E','S','W'};

        vector<int> dist(n*m, -1);
        auto id = [&](int r,int c){ return r*m + c; };

        queue<pair<int,int>> q;
        dist[id(start.first, start.second)] = 0;
        q.push(start);

        while (!q.empty()) {
            auto [r,c] = q.front(); q.pop();
            if (make_pair(r,c) == goal) break;
            for (int k=0;k<4;k++) {
                int nr = r+dr[k], nc = c+dc[k];
                if (nr<0||nr>=n||nc<0||nc>=m) continue;
                if (!can_move(r,c,nr,nc)) continue; // aqui entra checagem de paredes
                int v = id(nr,nc);
                if (dist[v] == -1) {
                    dist[v] = dist[id(r,c)] + 1;
                    parent[v] = id(r,c);
                    dir[v] = ch[k];
                    q.push({nr,nc});
                }
            }
        }

        // reconstrói
        string path;
        int v = id(goal.first, goal.second);
        if (dist[v]==-1) return ""; // sem caminho
        while (v != id(start.first,start.second)) {
            path.push_back(dir[v]);
            v = parent[v];
        }
        reverse(path.begin(), path.end());
        return path;
    }
};


No can_move, você verifica se não há parede entre as duas células (a partir dos dados de entrada do problema).

2.2 Dijkstra (SSSP com pesos ≥ 0)

Serve para:

Sending email (grafo esparso, até 20k nós, 50k arestas) 

763497

Dijkstra? (mesma pegada, ainda com reconstrução de caminho) 

765963

Full tank? (apenas com modelagem de estado) 

763497

Template bem padrão:

struct Dijkstra {
    int n;
    vector<vector<Edge>> g;
    Dijkstra(int n=0): n(n), g(n) {}

    void add_edge(int a, int b, ll w, bool undirected = true) {
        g[a].push_back({b,w});
        if (undirected) g[b].push_back({a,w});
    }

    // retorna dist; se parent != nullptr, também preenche parent
    vector<ll> run(int s, vector<int>* parent = nullptr) {
        vector<ll> dist(n, INFLL);
        if (parent) parent->assign(n, -1);

        using P = pair<ll,int>;
        priority_queue<P, vector<P>, greater<P>> pq;
        dist[s] = 0;
        pq.push({0,s});

        while (!pq.empty()) {
            auto [d,v] = pq.top(); pq.pop();
            if (d != dist[v]) continue;
            for (auto &e : g[v]) {
                if (dist[e.to] > d + e.w) {
                    dist[e.to] = d + e.w;
                    if (parent) (*parent)[e.to] = v;
                    pq.push({dist[e.to], e.to});
                }
            }
        }
        return dist;
    }

    // reconstrói caminho s -> t
    vector<int> get_path(int s, int t, const vector<int>& parent) {
        if (parent[t] == -1 && s != t) return {};
        vector<int> path;
        for (int v = t; v != -1; v = parent[v]) path.push_back(v);
        reverse(path.begin(), path.end());
        if (path.front() != s) return {};
        return path;
    }
};


Uso em “Sending email”:

Ler n,m,s,t, montar grafo, rodar dijkstra.run(s) e imprimir dist[t] ou "unreachable".

Uso em “Dijkstra?”:

Idem, mas com parent, depois get_path(0, n-1, parent) e imprimir os vértices 1-based.

2.3 Estado extra: Dijkstra em (cidade, combustível) – Full tank?

Ideia: o nó do grafo de estados é (city, fuel).

Estados: id = city * (Cmax+1) + fuel

Arestas:

Comprar 1 unidade de combustível: (city, f) -> (city, f+1) com custo preço[city], se f < C.

Viajar: se existe estrada city -> nxt com distância d e f >= d, então (city,f) -> (nxt, f-d) com custo 0.

Reaproveitando o Dijkstra, só muda como você monta o grafo. Um esqueleto:

ll full_tank(
    int n, int Cmax,
    const vector<int>& price,
    const vector<vector<pair<int,int>>>& roads,
    int s, int t
) {
    int N = n*(Cmax+1);
    Dijkstra dj(N);

    auto id = [&](int city,int fuel){ return city*(Cmax+1) + fuel; };

    for (int city=0; city<n; ++city) {
        for (int f=0; f<=Cmax; ++f) {
            int v = id(city,f);
            // comprar
            if (f < Cmax) {
                dj.add_edge(v, id(city,f+1), price[city], false);
            }
            // andar
            for (auto [to, d] : roads[city]) {
                if (f >= d) {
                    dj.add_edge(v, id(to, f-d), 0, false);
                }
            }
        }
    }

    auto dist = dj.run(id(s,0));
    ll ans = INFLL;
    for (int f=0; f<=Cmax; ++f) {
        ans = min(ans, dist[id(t,f)]);
    }
    return (ans == INFLL ? -1 : ans);
}

2.4 0-1 BFS (quando pesos são 0 ou 1)

Útil em vários contest problems (não necessariamente nesses PDFs, mas ótimo pra lib):
```
vector<int> bfs01(int n, int s, const vector<vector<Edge>>& g) {
    deque<int> dq;
    vector<int> dist(n, INT_MAX);
    dist[s] = 0;
    dq.push_front(s);
    while (!dq.empty()) {
        int v = dq.front(); dq.pop_front();
        for (auto &e : g[v]) {
            int nd = dist[v] + (int)e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                if (e.w == 0) dq.push_front(e.to);
                else dq.push_back(e.to);
            }
        }
    }
    return dist;
}
```
