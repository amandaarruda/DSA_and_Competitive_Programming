#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;


int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    

    int n;
    cin >> n;

    vector<int> distinct_numbers(n);
    
    for (int i = 0; i < n; i++) {
        cin >> distinct_numbers[i];
    }

    sort(distinct_numbers.begin(), distinct_numbers.end());

    distinct_numbers.erase(unique(distinct_numbers.begin(), distinct_numbers.end()), distinct_numbers.end());
    
    cout << distinct_numbers.size();
}
