
---
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX 50
  
typedef struct Command {
    int Play_Process;
    int *resource;
    struct Command *next;
} Command;
  
typedef struct Queue {
    Command *front, *rear;
    int Q_count;
} Queue;
  
void enQueue(Queue *q, Command item, int R_num) {
    int i;
    Command *temp = (Command*)malloc(sizeof(Command));
    temp->resource = (int*)malloc(sizeof(int)*R_num);
  
    temp->Play_Process = item.Play_Process;
    for(i = 0 ; i < R_num ; i++) {
        temp->resource[i] = item.resource[i];
    }
  
    temp->next = NULL;
  
    if(q->Q_count == 0) {
        q->front = temp;
        q->rear = temp;
    }
    else {
        q->rear->next = temp;
        q->rear = temp;
    }
    q->Q_count++;
}
  
Command deQueue(Queue *q, int R_num) {
    Command *temp = q->front;
    Command item;
    int i;
  
    item.resource = (int*)malloc(sizeof(int)*R_num);
    item.Play_Process = temp->Play_Process;
    q->front = q->front->next;
  
    for(i = 0 ; i < R_num ; i++) {
        item.resource[i] = temp->resource[i];
    }
    q->Q_count--;
    return item;
}
  
void inti_Q(Queue *q) {
    q->front = NULL;
    q->rear = NULL;
    q->Q_count = 0;
}
  
int Av_Re(int Av[], int Re_Ne[], int c) {
    int i;
    for(i = 0 ; i < c ; i++) {
        if(Re_Ne[i] > Av[i])
            return 0;
    }
    return 1;
}
  
void Array_copy1(int temp[], int origin[], int num) {
    int i;
    for(i = 0 ; i < num ; i++) {
        temp[i] = origin[i];
    }
}
  
void Array_copy2(int (*temp)[50], int (*origin)[50], int P_num, int R_num) {
    int i, j;
    for(i = 0 ; i < P_num ; i++) {
        for(j = 0 ; j < R_num ; j++) {
            temp[i][j] = origin[i][j];
        }
    }
}
  
int Request_Check(int (*All)[50], int (*Ne)[50], int Av[], Command Re, int P_num, int R_num) {
    int i, j;
    int All_temp[MAX][MAX];
    int N_temp[MAX][MAX];
    int Av_temp[MAX];
    int flag[MAX];
    int Next = 1;
  
    for(i = 0 ; i < P_num ; i++) {
        flag[i] = 0;
    }
  
    if(Av_Re(Ne[Re.Play_Process], Re.resource, R_num) == 0) {
        return 2;
    }
  
    Array_copy2(All_temp, All, P_num, R_num);
    Array_copy2(N_temp, Ne, P_num, R_num);
    Array_copy1(Av_temp, Av, R_num);
  
    for(i = 0 ; i < R_num ; i++) {
        N_temp[Re.Play_Process][i] -= Re.resource[i];
        Av_temp[i] -=  Re.resource[i];
    }
    for(i = 0 ; i < R_num ; i++) {
        All_temp[Re.Play_Process][i] += Re.resource[i];
    }
  
    while(Next) {
        Next = 0;
        for(i = 0 ; Next == 0 && i < P_num ; i++) {
            if(flag[i] == 0) {
                if(Av_Re(Av_temp, N_temp[i], R_num) == 0) {
                }
                else {
                    Next = 1;
                    for(j = 0 ; j < R_num ; j++)
                        Av_temp[j] += All_temp[i][j];
                    flag[i] = 1;
                }
            }
        }
    }
  
    for(i = 0 ; i < P_num ; i++) {
        if(flag[i] == 0) {
            return 0;
        }
    }
  
    for(i = 0 ; i < R_num ; i++) {
        Ne[Re.Play_Process][i] -= Re.resource[i];
        Av[i] -=  Re.resource[i];
    }
    for(i = 0 ; i < R_num ; i++) {
        All[Re.Play_Process][i] += Re.resource[i];
    }
    return 1;
}
  
void Release(int (*All)[50], int (*Ne)[50], int Av[], Command Re, int P_num, int R_num) {
    int i;
    for(i = 0 ; i < R_num ; i++) {
        All[Re.Play_Process][i] -= Re.resource[i];
    }
    for(i = 0 ; i < R_num ; i++) {
        Ne[Re.Play_Process][i] += Re.resource[i];
        Av[i] += Re.resource[i];
    }
}
  
int main() {
    int Process_num, Resource_num;
    int Resource_MAX[MAX];
    int Available[MAX];
    int Allocated[MAX][MAX];
    int Max[MAX][MAX];
    int Need[MAX][MAX];
  
    int i, j, temp, Q_count;
    char word[8];
  
    Queue Wait_Queue;
    Command C_temp;
  
    FILE *file = fopen("banker.inp", "rt");
    FILE *file2 = fopen("banker.out", "wt");
    fscanf(file, "%d%d", &Process_num, &Resource_num);
    inti_Q(&Wait_Queue);
  
    for(i = 0; i < Resource_num; i++) {
        fscanf(file, "%d", &Resource_MAX[i]);
    }
  
    for(i = 0; i < Process_num ; i++) {
        for(j = 0; j < Resource_num ; j++) {
            fscanf(file, "%d", &Max[i][j]);
        }
    }
  
    for(i = 0; i < Process_num ; i++) {
        for(j = 0; j < Resource_num ; j++) {
            fscanf(file, "%d", &Allocated[i][j]);
        }
    }
  
    for(i = 0; i < Process_num ; i++) {
        for(j = 0; j < Resource_num ; j++) {
            Need[i][j] = Max[i][j] - Allocated[i][j];
        }
    }
  
    for(i = 0; i < Resource_num; i++) {
        temp = 0;
        for(j = 0; j < Process_num; j++) {
            temp += Allocated[j][i];
        }
        Available[i] = Resource_MAX[i] - temp;
    }
  
    C_temp.resource = (int*)malloc(sizeof(int)*Resource_num);
  
    while(1) {
        fscanf(file, "%s", &word);
        fscanf(file, "%d", &C_temp.Play_Process);
        for(i = 0; i < Resource_num ; i++) {
            fscanf(file, "%d", &C_temp.resource[i]);
        }
  
        if(strcmp(word, "request") == 0) {
            if(Av_Re(Available, C_temp.resource, Resource_num)) {
                temp = Request_Check(Allocated, Need, Available, C_temp, Process_num, Resource_num);
                if(temp == 1 || temp == 2) {
                }
                else {
                    enQueue(&Wait_Queue, C_temp, Resource_num);
                }
  
                for(i = 0; i < Resource_num ; i++) {
                    printf("%d ", Available[i]);
                    fprintf(file2, "%d ", Available[i]);
                }
                printf("\n");
                fprintf(file2, "\n");
            }
            else {
                enQueue(&Wait_Queue, C_temp, Resource_num);
                for(i = 0; i < Resource_num ; i++) {
                    printf("%d ", Available[i]);
                    fprintf(file2, "%d ", Available[i]);
                }
                printf("\n");
                fprintf(file2, "\n");
            }
        }
        else if(strcmp(word, "release") == 0) {
            int Q_count_temp;
            Release(Allocated, Need, Available, C_temp, Process_num, Resource_num);
  
            Q_count_temp = Wait_Queue.Q_count;
            for(i = 0 ; i < Q_count_temp ; i++) {
                C_temp = deQueue(&Wait_Queue, Resource_num);
                if(Av_Re(Available, C_temp.resource, Resource_num) == 0) {
                    enQueue(&Wait_Queue, C_temp, Resource_num);
                    continue;
                }
                temp = Request_Check(Allocated, Need, Available, C_temp, Process_num, Resource_num);
                if(temp == 0) {
                    enQueue(&Wait_Queue, C_temp, Resource_num);
                }
                else {
                }
            }
  
            for(i = 0; i < Resource_num ; i++) {
                fprintf(file2, "%d ", Available[i]);
                printf("%d ", Available[i]);
            }
            printf("\n");
            fprintf(file2, "\n");
        }
        else
            break;
  
        C_temp.Play_Process = -1;
        for(i = 0 ; i < Resource_num ; i++)
            C_temp.resource[i] = -1;
    }
  
    fclose(file);
    fclose(file2);
    return 0;
}
```