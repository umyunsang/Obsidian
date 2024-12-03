
---
```c
#include <stdio.h>
#include <stdlib.h>
  
#define MAX_FRAMES 100
#define MAX_REFERENCES 10000
  
typedef struct {
    int page;
    int last_used; // LRU를 위한 마지막 사용 시간
    int future_use; // OPT를 위한 미래 사용 시간
} Frame;
  
// FIFO 알고리즘
int FIFO(int frames[], int frame_count, int ref[], int ref_count) {
    int page_faults = 0, next_replace = 0;
  
    for (int i = 0; i < ref_count; i++) {
        int found = 0;
        // 페이지가 프레임 안에 있는지 확인
        for (int j = 0; j < frame_count; j++) {
            if (frames[j] == ref[i]) {
                found = 1;
                break;
            }
        }
        // 페이지 부재 발생
        if (!found) {
            frames[next_replace] = ref[i];
            next_replace = (next_replace + 1) % frame_count; // FIFO 큐의 순

            page_faults++;
        }
    }
  
    return page_faults;
}
  
// LRU 알고리즘
int LRU(Frame frames[], int frame_count, int ref[], int ref_count) {
    int page_faults = 0, time = 0;
  
    for (int i = 0; i < ref_count; i++) {
        time++;
        int found = -1;
  
        // 페이지가 프레임 안에 있는지 확인
        for (int j = 0; j < frame_count; j++) {
            if (frames[j].page == ref[i]) {
                found = j;
                break;
            }
        }
  
        if (found != -1) {
            frames[found].last_used = time; // 사용 시간 갱신
        } else {
            // 페이지 부재 처리
            int lru_index = 0;
            for (int j = 1; j < frame_count; j++) {
                if (frames[j].last_used < frames[lru_index].last_used) {
                    lru_index = j;
                }
            }
            frames[lru_index].page = ref[i];
            frames[lru_index].last_used = time;
            page_faults++;
        }
    }
  
    return page_faults;
}
  
// OPT 알고리즘
int OPT(Frame frames[], int frame_count, int ref[], int ref_count) {
    int page_faults = 0;
  
    for (int i = 0; i < ref_count; i++) {
        int found = -1;
  
        // 페이지가 프레임 안에 있는지 확인
        for (int j = 0; j < frame_count; j++) {
            if (frames[j].page == ref[i]) {
                found = j;
                break;
            }
        }
  
        if (found == -1) {
            // OPT 교체 대상 찾기
            int replace_index = -1, farthest = i;
            for (int j = 0; j < frame_count; j++) {
                int next_use = -1;
                for (int k = i + 1; k < ref_count; k++) {
                    if (frames[j].page == ref[k]) {
                        next_use = k;
                        break;
                    }
                }
                if (next_use == -1) {
                    replace_index = j;
                    break;
                } else if (next_use > farthest) {
                    farthest = next_use;
                    replace_index = j;
                }
            }
            frames[replace_index].page = ref[i];
            page_faults++;
        }
    }
  
    return page_faults;
}
  
int main() {
    int frame_count, ref_count = 0;
    int ref[MAX_REFERENCES];
    int fifo_frames[MAX_FRAMES];
    Frame lru_frames[MAX_FRAMES];
    Frame opt_frames[MAX_FRAMES];
  
    // 입력 파일 읽기
    FILE *input = fopen("page.inp", "r");
    if (!input) {
        printf("Error opening input file.\n");
        return 1;
    }
    fscanf(input, "%d", &frame_count);
  
    while (fscanf(input, "%d", &ref[ref_count]) != EOF && ref[ref_count] != -1) {
        ref_count++;
    }
    fclose(input);
  
    // 초기화
    for (int i = 0; i < frame_count; i++) {
        fifo_frames[i] = -1;
        lru_frames[i] = (Frame){-1, 0, 0};
        opt_frames[i] = (Frame){-1, 0, 0};
    }
  
    // 각 알고리즘 실행
    int fifo_faults = FIFO(fifo_frames, frame_count, ref, ref_count);
    int lru_faults = LRU(lru_frames, frame_count, ref, ref_count);
    int opt_faults = OPT(opt_frames, frame_count, ref, ref_count);
  
    // 출력 파일 작성
    FILE *output = fopen("page.out", "w");
    if (!output) {
        printf("Error opening output file.\n");
        return 1;
    }
  
    fprintf(output, "FIFO: %d\n", fifo_faults);
    fprintf(output, "LRU: %d\n", lru_faults);
    fprintf(output, "OPT: %d\n", opt_faults);
  
    fclose(output);
    return 0;
}
```