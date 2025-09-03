
---
```c
#include <stdio.h>
#include <stdlib.h>
  
typedef struct {
    int arrival_time;
    int cpu_time;
    int waiting_time;
    int process_id;
} Process;
  
int compare(const void *a, const void *b) {
    Process *processA = (Process *)a;
    Process *processB = (Process *)b;
  
    if (processA->cpu_time == processB->cpu_time) {
        return processA->process_id - processB->process_id;
    }
    return processA->cpu_time - processB->cpu_time;
}
  
int main() {
    FILE *input_file = fopen("sjf.inp", "r");
    FILE *output_file = fopen("sjf.out", "w");
  
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
        processes[i].process_id = i;
    }
  
    int current_time = 0;
    int total_waiting_time = 0;
    int completed_processes = 0;
    int is_completed[n];
    for (int i = 0; i < n; i++) {
        is_completed[i] = 0;
    }
  
    while (completed_processes < n) {
        Process ready_queue[n];
        int ready_count = 0;
  
        for (int i = 0; i < n; i++) {
            if (processes[i].arrival_time <= current_time && !is_completed[i]) {
                ready_queue[ready_count++] = processes[i];
            }
        }
  
        if (ready_count > 0) {
            qsort(ready_queue, ready_count, sizeof(Process), compare);
  
            Process current_process = ready_queue[0];
  
            current_process.waiting_time = current_time - current_process.arrival_time;
            total_waiting_time += current_process.waiting_time;
  
            current_time += current_process.cpu_time;
            is_completed[current_process.process_id] = 1;
            completed_processes++;
        } else {
            current_time++;
        }
    }
  
    fprintf(output_file, "%d\n", total_waiting_time);
  
    fclose(input_file);
    fclose(output_file);
    return 0;
}
```