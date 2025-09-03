
---
```c
#include <stdio.h>
#include <stdlib.h>
#define MAX 1000
  
//구조체 선언
typedef struct Process {
     int Process_num; //프로세스 번호
     int Start_time; //시작시간
     int Work_time; //작동시간
     int End_time; //종료시간(바로 시작시 Start + Work & 큐에서 나올 경우 그 때 시간 + Work_time
     int Process_Size; //프로세스의 크기
     int Flag; //0 : 할당x, 1 : 할당o, 2 : 종료
     Process* Next; //큐를 만들기 위해 필요한 포인터
} Process;
  
 typedef struct Memory_Status {
     int Adress; //주소
     int PID_Process_num; //프로세스번호
     int Process_Size; //Process_Size;
     Memory_Status *R_link, *L_link;
} Memory_Status;
//구조체선언 끝
  
//큐관련
typedef struct Queue {
    Process *front, *rear;
    int Queue_Count;
} Queue;
void Queue_inti(Queue* q) { //큐 초기화
     q->front = NULL;
     q->rear = NULL;
     q->Queue_Count = 0;
}
  
void Queue_enQ(Queue* q, Process P) { //큐 삽입
    Process *temp = (Process*)malloc(sizeof(Process));
    temp->End_time = P.End_time;
    temp->Flag = P.Flag;
    temp->Process_num = P.Process_num;
    temp->Process_Size = P.Process_Size;
    temp->Start_time = P.Start_time;
    temp->Work_time = P.Work_time;
  
    if(q->Queue_Count == 0) {
        q->front = temp;
        q->rear = temp;
    }
    else {
        q->rear->Next = temp;
        q->rear = temp;
    }
  
    q->Queue_Count++;
}
  
void temp_Queue_enQ(Queue* q, Process P) { //혹시나 싶어 만들어본 큐함수-_-;
    Process *temp = (Process*)malloc(sizeof(Process));
    temp->End_time = P.End_time;
    temp->Flag = P.Flag;
    temp->Process_num = P.Process_num;
    temp->Process_Size = P.Process_Size;
    temp->Start_time = P.Start_time;
    temp->Work_time = P.Work_time;
  
    if(q->Queue_Count == 0) {
        q->front = temp;
        q->rear = temp;
    }
    else {
        temp->Next = q->front;
        q->front = temp;
    }
    q->Queue_Count++;
}
  
Process Queue_deQ(Queue* q) { //큐 삭제, 데이터의 변경은 큐가 나온 뒤 확정이되면 변경이 됨
    Process *temp = q->front;
    Process item;
  
    item.End_time = temp->End_time;
    item.Flag = temp->Flag;
    item.Process_num = temp->Process_num;
    item.Process_Size = temp->Process_Size;
    item.Start_time = temp->Start_time;
    item.Work_time = temp->Work_time;
  
    q->front = q->front->Next;
    free(temp);
  
    q->Queue_Count--;
    return item;
}
//큐관련 끝
  
/////////리스트 관련 삽입 삭제, 초기화////////////
typedef struct two_List { //탐색을 위한 이중연결리스트 헤더구조체
    Memory_Status *Next;
} two_List;
  
void List_inti(two_List* L) {
    L->Next = NULL;
}
  
void List_1_insert(two_List* L) { //제일 처음 시작
    Memory_Status *temp = (Memory_Status*)malloc(sizeof(Memory_Status));
  
    L->Next = temp;
    temp->Adress = 0;
    temp->L_link = NULL;
    temp->R_link = NULL;
    temp->Process_Size = 1000;
    temp->PID_Process_num = -1;
    //printf("프로세스번호 : %d, 우측 주소 : %d\n", temp->PID_Process_num, temp->R_link);
}
  
void List_insert(two_List *L, Memory_Status* M, Process P) { //삽입
    Memory_Status *temp = (Memory_Status*)malloc(sizeof(Memory_Status));
  
    if(M->Process_Size == P.Process_Size) { //hole과 크기가 같은경우
        M->PID_Process_num = P.Process_num;
        //printf("%d가 주소 %d에 들어감, 같음\n", P.Process_num, M->Adress);
        free(temp);
    }
    else { //다른경우
        if(M->L_link == NULL) { //맨 처음일 경우
           L->Next = temp;
        }
        else {
            M->L_link->R_link = temp
        }
        temp->L_link = M->L_link;
        temp->R_link = M;
        M->L_link = temp;
  
        temp->Adress = M->Adress;
        temp->PID_Process_num = P.Process_num;
        temp->Process_Size = P.Process_Size;
        M->Adress += P.Process_Size;
        M->Process_Size -= P.Process_Size;
        //printf("%d가 주소 %d에 들어감\n", P.Process_num, temp->Adress);
    }
}
  
void List_Delete(two_List* L, Memory_Status* M) { //리스트에서 메모리 삭제 함수, free 확인해보기
    if(M->R_link == NULL) { //맨끝이 삭제 될 경우
        if(M->L_link->PID_Process_num == -1) { //좌측이hole인경우
            M->L_link->Process_Size += M->Process_Size;
            M->L_link->R_link = M->R_link;
  
            //free(M);
        }
        else { //좌측이 hole아닌 경우
            M->PID_Process_num = -1;
        }
    }
    else if(M->L_link == NULL) { //맨처음이 삭제 될경우
        if(M->R_link->PID_Process_num != -1) { //우측이 hole이 아닌 경우
            M->PID_Process_num = -1;
        }
        else {
            L->Next = M->R_link;
            M->R_link->Adress = M->Adress;
            M->R_link->Process_Size += M->Process_Size;
            M->R_link->L_link = M->L_link;
  
            //free(M);
        }
   }
    else { //중앙의 것이 삭제 될경우
        if(M->L_link->PID_Process_num != -1 && M->R_link->PID_Process_num != -1) { //양쪽이 둘 다 hole이 아닌 경우
            M->PID_Process_num = -1;
        }
        else if(M->L_link->PID_Process_num == -1 && M->R_link->PID_Process_num != -1) { //왼쪽만 hole인 경우
            M->L_link->Process_Size += M->Process_Size;
  
            M->R_link->L_link = M->L_link;
            M->L_link->R_link = M->R_link;
  
            //free(M);
        }
        else if(M->L_link->PID_Process_num != -1 && M->R_link->PID_Process_num == -1) { //오른쪽만 hole인 경우
            M->R_link->Adress = M->Adress;
            M->R_link->Process_Size += M->Process_Size;
  
            M->L_link->R_link = M->R_link;
            M->R_link->L_link = M->L_link;
  
            //free(M);
        }
        else { //둘다 hole인 경우
            M->L_link->Process_Size = M->Process_Size + M->R_link->Process_Size + M->L_link->Process_Size;
            M->L_link->R_link = M->R_link->R_link;
            if(M->R_link->R_link == NULL) {//양쪽 둘 다 홀인 상태에서 우측 홀 이후가 NULL인 경우
            }
            else {
                M->R_link->R_link->L_link = M->L_link;
            }
  
            //free(M->R_link);
            //free(M);
        }
    }
}
  
void List_Process_Delete(two_List* L, Process P) { //리스트와 프로세스번호 비교하고 삭제하는 함수
    Memory_Status *temp;
    for(temp = L->Next ; temp != NULL ; temp = temp->R_link) {
        if(temp->PID_Process_num == P.Process_num) {
            List_Delete(L, temp);
        }
    }
}
  
/////////리스트 관련 삽입 삭제, 초기화, 출력 완료///////
  
////////리스트 탐색 1. first////////////
int List_Search_First(two_List *L, Process P) { //성공 : 1, 실패 : 2
    Memory_Status *temp;
  
    for(temp = L->Next ; temp != NULL ; temp = temp->R_link) {
        if(temp->PID_Process_num == -1 && temp->Process_Size >= P.Process_Size) { //hole이고 P의 사이즈보다 크다면
            List_insert(L, temp, P); //리스트에 삽입
            return 1;
        }
    }
    return 2;
}
////////리스트 탐색 2. worst///////////
int List_Search_Worst(two_List *L, Process P) { //성공 : 1, 실패 : 2
    Memory_Status *temp;
    Memory_Status *temp2 = (Memory_Status*)malloc(sizeof(Memory_Status));
  
    temp2->Process_Size = 0;
    for(temp = L->Next ; temp != NULL ; temp = temp->R_link) {
        if(temp->PID_Process_num == -1 && temp->Process_Size >= P.Process_Size) { //hole이고 P의 사이즈보다 크다면
            if(temp->Process_Size > temp2->Process_Size){
                temp2 = temp;
            }
        }
    }
  
    if(temp2->Process_Size != 0) {
        List_insert(L, temp2, P); //리스트에 삽입
        //free(temp2);
        return 1;
    }
    else {
        return 2;
    }
}
////////리스트 탐색 3. best////////////
int List_Search_Best(two_List *L, Process P) { //성공 : 1, 실패 : 2
    Memory_Status *temp;
    Memory_Status *temp2 = (Memory_Status*)malloc(sizeof(Memory_Status));
  
    temp2->Process_Size = 1001;
  
    for(temp = L->Next ; temp != NULL ; temp = temp->R_link) {
        if(temp->PID_Process_num == -1 && temp->Process_Size >= P.Process_Size) { //hole이고 P의 사이즈보다 크다면
            if(temp->Process_Size < temp2->Process_Size){
                temp2 = temp;
            }
        }
    }
  
    if(temp2->Process_Size <1001) {
        List_insert(L, temp2, P); //리스트에 삽입
        return 1;
    }
    else {
        return 2;
    }
}
  
//프로세스리스트 복사 함수
void Process_copy(Process *Ori, Process *temp, Process *temp1, int cout) {
    int i;
    for(i = 0 ; i < cout ; i++) {
        temp[i].End_time = -1;
        temp[i].Flag = 0;
        temp[i].Next = NULL;
        temp[i].Process_num = Ori[i].Process_num;
        temp[i].Process_Size = Ori[i].Process_Size;
        temp[i].Start_time = Ori[i].Start_time;
        temp[i].Work_time = Ori[i].Work_time;
  
        temp1[i].End_time = -1;
        temp1[i].Flag = 0;
        temp1[i].Next = NULL;
        temp1[i].Process_num = Ori[i].Process_num;
        temp1[i].Process_Size = Ori[i].Process_Size;
        temp1[i].Start_time = Ori[i].Start_time;
        temp1[i].Work_time = Ori[i].Work_time;
    }
}
   
//출력값을 위한 탐색
Memory_Status* Result_Search(two_List *L, Process P) {
    Memory_Status *temp;
    for(temp = L->Next ; temp != NULL ; temp = temp->R_link) {
        if(temp->PID_Process_num == P.Process_num) {
            return temp;
        }
    }
}
  
int main() {
    two_List First_head;
    two_List Best_head;
    two_List Worst_head;
  
    Queue Wait_Queue;
    Queue Wait_Queue_Best;
    Queue Wait_Queue_Worst;
  
    Process Process_List[MAX];
    Process Process_List_Best[MAX];
    Process Process_List_Worst[MAX];
  
    int time;
    int Process_count;
    int i;
    FILE *file = fopen("allocation.inp", "rt");
    FILE *file1 = fopen("allocation.out", "wt");
  
    List_inti(&First_head);
    List_1_insert(&First_head);
    List_inti(&Best_head);
    List_1_insert(&Best_head);
    List_inti(&Worst_head);
    List_1_insert(&Worst_head);
    Queue_inti(&Wait_Queue);
    Queue_inti(&Wait_Queue_Best);
    Queue_inti(&Wait_Queue_Worst);
  
    fscanf(file, "%d", &Process_count); //첫줄 받기
    for(i = 0 ; i < Process_count ; i++) { //Process리스트받기
        Process_List[i].Process_num = i;
        fscanf(file, "%d %d %d", &Process_List[i].Start_time, &Process_List[i].Work_time, &Process_List[i].Process_Size);
        Process_List[i].Flag = 0;
        Process_List[i].End_time = -1;
        Process_List[i].Next = NULL;
    }
  
    Process_copy(Process_List, Process_List_Best, Process_List_Worst, Process_count);
  
    //여기서 first시작
    for(time = 0 ; ; time++) {
        int temp_del = 0;
        for(i = 0 ; i < Process_count ; i++) { //끝난 프로세스를 찾음
            if(Process_List[i].End_time == time) {
                List_Process_Delete(&First_head, Process_List[i]);
                Process_List[i].Flag = 2;
                //printf("%d초에 %d번 프로세스 종료! \n", time, Process_List[i].Process_num);
                temp_del++;//반환함
            }
        }
  
        //한개라도 반환한것이 있다면
        if(temp_del > 0) {
            int Q_count, temp;
            Process P_temp;
  
            Q_count = Wait_Queue.Queue_Count;
            for(i = 0 ; i < Q_count ; i++) {
                P_temp = Queue_deQ(&Wait_Queue);
                temp = List_Search_First(&First_head, P_temp);
                if(temp == 1) { //찾는 것이 성공하면
                        Process_List[P_temp.Process_num].Flag = 1;
                        Process_List[P_temp.Process_num].End_time = time + Process_List[P_temp.Process_num].Work_time;
                        //printf("%d초에 %d가 큐에서 나옴\n", time, P_temp.Process_num);
                }
                else { //실패하면
                    //printf("%d초에 %d가 큐에 들어감\n", time, P_temp.Process_num);
                    Queue_enQ(&Wait_Queue, P_temp);
                }
            }
        }
  
        for(i = 0 ; i < Process_count ; i++) { //시작프로세스를 찾음
            if(Process_List[i].Start_time == time) {
                int temp;
                temp = List_Search_First(&First_head, Process_List[i]); //First_fit으로 검색
                if(temp == 1) { //성공하면
                    Process_List[i].Flag = 1; //작동 중
                    Process_List[i].End_time = time + Process_List[i].Work_time; //종료시간 설정
                }
                else { //실패하면
                    Queue_enQ(&Wait_Queue, Process_List[i]);
                    //printf("%d초에 %d가 큐에 들어감\n", time, Process_List[i].Process_num);
                }
                //printf("%d초에 %d번 프로세스 시작! \n", time, Process_List[i].Process_num);
            }
        }
  
        //종료부분
        if(Process_List[Process_count - 1].Flag == 1) {
            Memory_Status *temp = Result_Search(&First_head, Process_List[Process_count-1]);
            printf("주소 : %d\n", temp->Adress);
            fprintf(file1, "%d\n", temp->Adress);
            break;
        }
    }
    //여기서 Best 시작
    for(time = 0 ; ; time++) {
        int temp_del = 0;
        for(i = 0 ; i < Process_count ; i++) { //끝난 프로세스를 찾음
            if(Process_List_Best[i].End_time == time) {
                List_Process_Delete(&Best_head, Process_List_Best[i]);
                Process_List_Best[i].Flag = 2;
                //printf("%d초에 %d번 프로세스 종료! \n", time, Process_List[i].Process_num);
                temp_del++;//반환함
            }
        }
  
        //한개라도 반환한것이 있다면
        if(temp_del >= 1) {
            int Q_count, temp;
            Process P_temp;
  
            Q_count = Wait_Queue_Best.Queue_Count;
            for(i = 0 ; i < Q_count ; i++) {
                P_temp = Queue_deQ(&Wait_Queue_Best);
                temp = List_Search_Best(&Best_head, P_temp);
                if(temp == 1) { //찾는 것이 성공하면
                        Process_List_Best[P_temp.Process_num].Flag = 1;
                        Process_List_Best[P_temp.Process_num].End_time = time + Process_List_Best[P_temp.Process_num].Work_time;
                        //printf("%d초에 %d가 큐에서 나옴\n", time, P_temp.Process_num);
                }
                else { //실패하면
                    //printf("%d초에 %d가 큐에 들어감\n", time, P_temp.Process_num);
                    Queue_enQ(&Wait_Queue_Best, P_temp);
                }
            }
        }
  
        for(i = 0 ; i < Process_count ; i++) { //시작프로세스를 찾음
            if(Process_List_Best[i].Start_time == time) {
                int temp;
                temp = List_Search_Best(&Best_head, Process_List_Best[i]); //Best_fit으로 검색
                if(temp == 1) { //성공하면
                    Process_List_Best[i].Flag = 1; //작동 중
                    Process_List_Best[i].End_time = time + Process_List_Best[i].Work_time; //종료시간 설정
                }
                else { //실패하면
                    Queue_enQ(&Wait_Queue_Best, Process_List_Best[i]);
                    //printf("%d초에 %d가 큐에 들어감\n", time, Process_List[i].Process_num);
                }
                //printf("%d초에 %d번 프로세스 시작! \n", time, Process_List[i].Process_num);
            }
        }
  
        //종료부분
        if(Process_List_Best[Process_count - 1].Flag == 1) {
            Memory_Status *temp = Result_Search(&Best_head, Process_List_Best[Process_count-1]);
            printf("주소 : %d\n", temp->Adress);
            fprintf(file1, "%d\n", temp->Adress);
            break;
        }
    }
  
    //Worst시작
        for(time = 0 ; ; time++) {
        int temp_del = 0;
  
        for(i = 0 ; i < Process_count ; i++) { //끝난 프로세스를 찾음
            if(Process_List_Worst[i].End_time == time) {
                List_Process_Delete(&Worst_head, Process_List_Worst[i]);
                Process_List_Worst[i].Flag = 2;
                //printf("%d초에 %d번 프로세스 종료! \n", time, Process_List[i].Process_num);
                temp_del++;//반환함
            }
        }
  
        //한개라도 반환한것이 있다면
        if(temp_del >= 1) {
            int Q_count, temp;
            Process P_temp;
            Q_count = Wait_Queue_Worst.Queue_Count;
            for(i = 0 ; i < Q_count ; i++) {
                P_temp = Queue_deQ(&Wait_Queue_Worst);
                temp = List_Search_Worst(&Worst_head, P_temp);
                if(temp == 1) { //찾는 것이 성공하면
                        Process_List_Worst[P_temp.Process_num].Flag = 1;
                        Process_List_Worst[P_temp.Process_num].End_time = time + Process_List_Worst[P_temp.Process_num].Work_time;
                        //printf("%d초에 %d가 큐에서 나옴\n", time, P_temp.Process_num);
                }
                else { //실패하면
                    //printf("%d초에 %d가 큐에 들어감\n", time, P_temp.Process_num);
                    Queue_enQ(&Wait_Queue_Worst, P_temp);
                }
            }
        }
  
        for(i = 0 ; i < Process_count ; i++) { //시작프로세스를 찾음
            if(Process_List_Worst[i].Start_time == time) {
                int temp;
                temp = List_Search_Worst(&Worst_head, Process_List_Worst[i]); //Worst_fit으로 검색
                if(temp == 1) { //성공하면
                    Process_List_Worst[i].Flag = 1; //작동 중
                    Process_List_Worst[i].End_time = time + Process_List_Worst[i].Work_time; //종료시간 설정
                }
                else { //실패하면
                    Queue_enQ(&Wait_Queue_Worst, Process_List_Worst[i]);
                    //printf("%d초에 %d가 큐에 들어감\n", time, Process_List[i].Process_num);
                }
                //printf("%d초에 %d번 프로세스 시작! \n", time, Process_List[i].Process_num);
            }
        }
  
        //종료부분
        if(Process_List_Worst[Process_count - 1].Flag == 1) {
            Memory_Status *temp = Result_Search(&Worst_head, Process_List_Worst[Process_count-1]);
            printf("주소 : %d\n", temp->Adress);
            fprintf(file1, "%d\n", temp->Adress);
            break;
        }
    }
  
    fclose(file);
    fclose(file1);
    return 0;
}
```