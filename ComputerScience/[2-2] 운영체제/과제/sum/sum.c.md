
---
```c
#include <stdio.h>

int main() {
    FILE *fp, *fpOut;
    int i, j, k, min, max, n, sum;
  
    fp = fopen("sum.inp", "r");
  
    fpOut = fopen("sum.out", "w");
  
    fscanf(fp, "%d", &n);
  
    for (i = 0; i < n; i++) {
        fscanf(fp, "%d %d", &j, &k);
  
        if (j < k) {
            min = j;
            max = k;
        } else {
            min = k;
            max = j;
        }

        sum = 0;
        for (int x = min; x <= max; x++) {
            sum += x;
        }
  
        fprintf(fpOut, "%d\n", sum);
    }

    fclose(fp);
    fclose(fpOut);
  
    return 0;
}
```