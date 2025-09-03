
---
```c
#include <stdio.h>
#include <stdlib.h>
  
typedef struct {
    int arrival_time;
    int cpu_time;
    int remaining_time;
    int waiting_time;
    int process_id;
} Process;
  
int main() {
    FILE *input_file = fopen("srtf.inp", "r");
    FILE *output_file = fopen("srtf.out", "w");
  
    if (!input_file || !output_file) {
        perror("File opening failed");
        return 1;
    }
  
    int n;
    fscanf(input_file, "%d", &n);
    Process processes[n];
  
    for (int i = 0; i < n; i++) {
        fscanf(input_file, "%d %d", &processes[i].arrival_time, &processes[i].cpu_time);
        processes[i].remaining_time = processes[i].cpu_time;
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
        Process *current_process = NULL;
  
        for (int i = 0; i < n; i++) {
            if (processes[i].arrival_time <= current_time && !is_completed[i]) {
                if (current_process == NULL || processes[i].remaining_time < current_process->remaining_time ||
                    (processes[i].remaining_time == current_process->remaining_time && processes[i].process_id < current_process->process_id)) {
                    current_process = &processes[i];
                }
            }
        }
  
        if (current_process != NULL) {
            current_process->remaining_time--;
            current_time++;
  
            for (int i = 0; i < n; i++) {
                if (!is_completed[i] && &processes[i] != current_process && processes[i].arrival_time <= current_time) {
                    processes[i].waiting_time++;
                }
            }
  
            if (current_process->remaining_time == 0) {
                is_completed[current_process->process_id] = 1;
                completed_processes++;
                total_waiting_time += current_time - current_process->arrival_time - current_process->cpu_time;
            }
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