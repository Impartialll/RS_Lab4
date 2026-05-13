#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>
#include <algorithm>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

void Task1() {
    int m = 8;
    int n = 15;
    vector<vector<int>> A(m, vector<int>(n));
    srand(time(NULL));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 101 - 50;
        }
    }

    omp_lock_t file_lock;
    omp_init_lock(&file_lock);
    ofstream out("results.txt");

    omp_set_num_threads(m);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int sum = 0;
        for (int j = 0; j < n; j++) {
            if (A[tid][j] > 0) {
                sum += A[tid][j];
            }
        }

        bool failed_logged = false;
        while (!omp_test_lock(&file_lock)) {
            if (!failed_logged) {
#pragma omp critical
                {
                    out << "Потік " << tid << " - невдала спроба увійти до закритої секції.\n";
                }
                failed_logged = true;
            }
        }

        out << "Початок закритої секції... Потік " << tid << "\n";
        out << "Результат рядка " << tid << ": " << sum << "\n";
        out << "Кінець закритої секції... Потік " << tid << "\n";

        omp_unset_lock(&file_lock);
    }

    omp_destroy_lock(&file_lock);
    out.close();

    cout << "Результати успішно записані у файл results.txt" << endl;
}

double f(double x) {
    return x / (sin(x) * sin(x));
}

void Task2() {
    double a = M_PI / 4.0;
    double b = M_PI / 3.0;
    int N = 20000000;
    double h = (b - a) / N;

    double start_seq = omp_get_wtime();
    double sum_seq = 0.0;
    for (int i = 1; i < N; i++) {
        sum_seq += f(a + i * h);
    }
    double I_seq = h * ((f(a) + f(b)) / 2.0 + sum_seq);
    double t_seq = omp_get_wtime() - start_seq;

    cout << fixed << setprecision(6);
    cout << "Послідовний результат: " << I_seq << " (Час: " << t_seq << " с)\n\n";

    cout << setw(10) << "\tПотоки" << setw(15) << "\t\tРозклад" << setw(10) << "Chunk"
        << setw(15) << "Час (с)" << setw(15) << "\tПрискорення" << "\n";
    cout << string(65, '-') << "\n";

    vector<int> thread_counts = { 2, 4, 8 };
    int chunk = 1000; 

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);

        double start_par = omp_get_wtime();
        double sum_par = 0.0;
#pragma omp parallel for reduction(+:sum_par) schedule(static, 1000)
        for (int i = 1; i < N; i++) {
            sum_par += f(a + i * h);
        }
        double t_par = omp_get_wtime() - start_par;
        cout << setw(10) << threads << setw(15) << "static" << setw(10) << chunk
            << setw(15) << t_par << setw(15) << (t_seq / t_par) << "\n";

        
        start_par = omp_get_wtime();
        sum_par = 0.0;
#pragma omp parallel for reduction(+:sum_par) schedule(dynamic, 1000)
        for (int i = 1; i < N; i++) {
            sum_par += f(a + i * h);
        }
        t_par = omp_get_wtime() - start_par;
        cout << setw(10) << threads << setw(15) << "dynamic" << setw(10) << chunk
            << setw(15) << t_par << setw(15) << (t_seq / t_par) << "\n";

        
        start_par = omp_get_wtime();
        sum_par = 0.0;
#pragma omp parallel for reduction(+:sum_par) schedule(guided, 1000)
        for (int i = 1; i < N; i++) {
            sum_par += f(a + i * h);
        }
        t_par = omp_get_wtime() - start_par;
        cout << setw(10) << threads << setw(15) << "guided" << setw(10) << chunk
            << setw(15) << t_par << setw(15) << (t_seq / t_par) << "\n";
    }
}

void Task3() {
    int N = 20000;
    vector<int> arr1(N);
    vector<int> arr2(N);

    for (int i = 0; i < N; i++) {
        int val = rand() % 100000;
        arr1[i] = val;
        arr2[i] = val;
    }

    double start_seq = omp_get_wtime();
    for (int i = 0; i < N - 1; i++) {
        for (int j = 0; j < N - i - 1; j++) {
            if (arr1[j] > arr1[j + 1]) {
                swap(arr1[j], arr1[j + 1]);
            }
        }
    }
    double t_seq = omp_get_wtime() - start_seq;

    omp_set_num_threads(omp_get_max_threads());
    double start_par = omp_get_wtime();
    for (int phase = 0; phase < N; phase++) {
        if (phase % 2 == 0) {
#pragma omp parallel for
            for (int i = 0; i < N - 1; i += 2) {
                if (arr2[i] > arr2[i + 1]) {
                    swap(arr2[i], arr2[i + 1]);
                }
            }
        }
        else {
#pragma omp parallel for
            for (int i = 1; i < N - 1; i += 2) {
                if (arr2[i] > arr2[i + 1]) {
                    swap(arr2[i], arr2[i + 1]);
                }
            }
        }
    }
    double t_par = omp_get_wtime() - start_par;

    cout << fixed << setprecision(6);
    cout << "Розмір масиву: " << N << "\n";
    cout << "Час послідовного сортування: " << t_seq << " с\n";
    cout << "Час паралельного сортування: " << t_par << " с\n";
    cout << "Прискорення: " << t_seq / t_par << "\n";
}

int main() {
    cout << "Завдання 1: Матриця та блокування\n";
    Task1();

    cout << "\nЗавдання 2: Метод трапецій\n";
    Task2();

    cout << "\nЗавдання 3: Метод парної-непарної перестановки\n";
    Task3();

    return 0;
}