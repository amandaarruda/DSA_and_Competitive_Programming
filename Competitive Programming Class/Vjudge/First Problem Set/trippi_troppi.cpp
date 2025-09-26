#include <iostream>
#include <string>

using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; 
    cin >> n;

    int i;
    for (i = 0; i < n; i++){
        string phrase;
        getline(cin >> ws, phrase);

        cout << phrase[0];
        int j;
        for (int j = 0; j < (int)phrase.size(); j++) {
            if (phrase[j - 1] == ' ') { 
                cout << phrase[j];
            }
        }
        cout << "\n";
    }
}
