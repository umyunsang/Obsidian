
---
```c
#include <stdio.h>
  
typedef struct {
    int arrival_time;
    int cpu_time;
    int waiting_time;
} Process;
  
int main() {
    FILE *input_file = fopen("fcfs.inp", "r");
    FILE *output_file = fopen("fcfs.out", "w");
  
    if (!input_file || !output_file) {
        perror("File opening failed");
        return 1;
    }
  
    int n;
    fscanf(input_file, "%d", &n);
    Process processes[n];
  
    for (int i = 0; i < n; i++) {
        fscanf(input_file, "%d %d", &processes[i].arrival_time, &processes[i].cpu_time);
        processes[i].waiting_time = 0;
    }
  
    int current_time = 0;
    int total_waiting_time = 0;
  
    // FCFS 스케줄링
    for (int i = 0; i < n; i++) {
        if (current_time < processes[i].arrival_time) {
            current_time = processes[i].arrival_time;
        }
        processes[i].waiting_time = current_time - processes[i].arrival_time;
        total_waiting_time += processes[i].waiting_time;
        current_time += processes[i].cpu_time;
    }
  
    fprintf(output_file, "%d\n", total_waiting_time);
    fclose(input_file);
    fclose(output_file);
    return 0;
}
```