// auto é uma palavra-chave que deixa o compilador deduzir automaticamente o tipo da variável 
// a partir do valor de inicialização.

#include <bits/stdc++.h>
using namespace std;

int main() {
    multiset<int> s = {1, 2, 4, 4, 5, 7};

    cout << "Conjunto: ";
    for (int x : s) cout << x << " ";
    cout << "\n";

    int valor = 4;

    // auto evita escrever "multiset<int>::iterator"
    auto it1 = s.lower_bound(valor); // primeiro >= 4
    auto it2 = s.upper_bound(valor); // primeiro > 4

    if (it1 != s.end()) // "sentinela" que marca o fim do container. se == s.end(), chegou ao fim
        cout << "lower_bound(" << valor << ") = " << *it1 << "\n";
    if (it2 != s.end())
        cout << "upper_bound(" << valor << ") = " << *it2 << "\n";

    return 0;
}

// Conjunto: 1 2 4 4 5 7 
// lower_bound(4) = 4
// upper_bound(4) = 5
