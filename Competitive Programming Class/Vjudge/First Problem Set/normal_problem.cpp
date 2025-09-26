#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; // test cases
    cin >> n;

    do{
        string s;
        cin >> s;

        string mirrored_string = "";

        for (int i = 0; i < (int)s.size(); i++) {
            if (s[i] == 'q') {
                mirrored_string += "p";
            } else if (s[i] == 'p') {
                mirrored_string += "q";
            } else {
                mirrored_string += "w";
            }
        }
        reverse(mirrored_string.begin(), mirrored_string.end());
        cout << mirrored_string << "\n";
    }while(--n);
}
