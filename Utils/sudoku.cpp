#include <bits/stdc++.h>
using namespace std;

bool isValidSudoku(vector<vector<int>>& board, int n) {
    int k = sqrt(n); // tamanho do subquadrado

    // conjuntos para rastrear duplicatas
    vector<unordered_set<int>> rows(n), cols(n);
    map<pair<int, int>, unordered_set<int>> squares;

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            int val = board[r][c];

            // valor fora do intervalo 1..n
            if (val < 1 || val > n) return false;

            pair<int, int> squareKey = {r / k, c / k};

            // verificar duplicatas
            if (rows[r].count(val) || cols[c].count(val) || squares[squareKey].count(val))
                return false;

            // registrar valor
            rows[r].insert(val);
            cols[c].insert(val);
            squares[squareKey].insert(val);
        }
    }
    return true;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;

    while (t--) {
        int n;
        cin >> n;
        vector<vector<int>> board(n, vector<int>(n));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                cin >> board[i][j];

        cout << (isValidSudoku(board, n) ? "yes" : "no") << "\n";
    }
}
