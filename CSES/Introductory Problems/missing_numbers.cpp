#include <iostream>

using namespace std;

int main()
{
    long int n;
    cin >> n;

    long int expectedSum = (((1+n)*n)/2);

    long int i;

    long int actualSum = 0;

    for(i = 0; i < n-1; i++)
    {
        long int v;
        cin >> v;
        actualSum = actualSum + v;
    }

    long int result = expectedSum - actualSum;
    cout << result;
}
