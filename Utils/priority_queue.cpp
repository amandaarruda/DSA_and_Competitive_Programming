#include <functional>
#include <queue>
#include <vector>
#include <iostream>

template<typename T> void print_queue(T& q) {
    while(!q.empty()) {
        std::cout << q.top() << " ";
        q.pop();
    }
    std::cout << '\n';
}

int main()
{
    std::priority_queue<int> q; // max heap
    for(int elm : {1,8,5,6,3,4,0,9,7,2}) { q.push(elm); }
    print_queue(q);

    std::priority_queue<int, std::vector<int>, std::greater<int> > q2; // min heap
    for(int elm : {1,8,5,6,3,4,0,9,7,2}) { q2.push(elm); }
    print_queue(q2);

    // Using lambda to compare elements.
    auto cmp = [](int left, int right) { return (left) > (right); }; // max heap
    std::priority_queue<int, std::vector<int>, decltype(cmp)> q3(cmp);

    for(int elm : {1,8,5,6,3,4,0,9,7,2}) { q3.push(elm); }
    print_queue(q3);
}
