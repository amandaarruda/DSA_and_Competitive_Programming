// TOPIC: Map in C++
// keys unicas, values podem se repetir

// NOTES:
// 1. Syntax: map<T1, T2> obj;  // where T1 is key type and T2 is value type.
// 2. std::map is an associative container that stores elements in key-value combination
//    where key should be unique, otherwise it overrides the previous value.
// 3. It is implemented using Self-Balance Binary Search Tree (AVL/Red-Black Tree).
// 4. It stores key-value pairs in sorted order on the basis of key (ascending/descending).
// 5. std::map is generally used in Dictionary type problems.

// EXAMPLE: Dictionary

#include <iostream>
#include <map>
#include <functional>
#include <vector>
using namespace std;

int main() {
    std::map<string, int> Map;
    Map["Chotu"] = 909090909;
    Map["Amit"] = 982349819;
    Map.insert(std::make_pair("Bot", 782348818));

    // Loop through map
    for (auto &el : Map) {
        cout << el.first << " " << el.second << endl;
    }

    // Access using [] operator
    cout << Map["Chotu"] << endl;

    map<string, vector<int>> Map;

    // Inserção de valores
    Map["Chotu"].push_back(90909009);
    Map["Amit"].push_back(234123413);
    Map["Amit"].push_back(2343413);

    // Loop pelo map
    for (auto &el1 : Map) {
        cout << el1.first << endl; // imprime a chave
        for (auto &el2 : el1.second) { // percorre o vector<int>
            cout << el2 << " ";
        }
        cout << endl;
    }

    return 0;
}
