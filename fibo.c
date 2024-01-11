#include <stdio.h>
long int powerN(int num, int n)
{
    if (n==0)
      return 1;
    if(n==1)
      return num;
    if(n%2==0)
    {
        long int result = powerN(num, n/2);
        return result*result;
    }
    else
    {
        long int result = powerN(num, (n-1)/2);
        return result*result*num;
    }
}
int main()
{
    printf("%ld\n", powerN(2,47));
    return 0;
}