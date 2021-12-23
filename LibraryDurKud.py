"""

--------------------ФУНКЦИИ, ДОСТУПНЫЕ ПОЛЬЗОВАТЕЛЮ--------------------

func(x) - Функция-декораттор питоновского eval - считывает строку от x и выполняет её

def diff_my(fx ,x0, dx): - Функция вычисления производной в точке x0 методом двусторонней разности
    fx - eval - строка, представляющая собой функцию
    x0 - float - Точка в которой ищется производная методом двусторонней разности
    dx - float - Приращение аргумента по которому считается формула

    Возвращает float - значение производной, посчитаной по формуле
def diff_std(f, x, x0):
    Фунция вычисления производной в точке питоновской библиотекой
    f - eval - строка, представляющая собой функцию
    x - переменная по которой происходит дифференцирование
    x0 - float - Точка в которой ищется производная методом двусторонней разности

    Возвращает float - значение производной, посчитаной по формуле

def inspection(a,b): # Проверка на ОДЗ 
    Процедура изменения двух массивов, где значения одного массива принимаются
    за корректные. Массив с некорректными значениями заменяет свои значения на корректные из первого массива
    
    a - list - массив с корректными значениями
    b - list - массив с проверяемыми на некорректность значениями      

def xrep(x, st): 
    Функция замены зануляемых интерпретатором значений на строки
    x - folat - точка, котороя подставляется в функцию
    st - str - строка функции

    Возвращает строку - выражение, которое не может корректно посчитать интерпретатор
    но наглядное для пользователя

def square_integral(f, left, right)
    Фукнция подсчёта площади под функцией через интеграл
    с помощью библиотеки sypmy методом integrate

    f - eval - строка содежащая функцию
    left - float - левая точка интервала. Точка от которой считается площаль
    right - float - правая точка интервала. Точка на которой заканчивается интервал и подсчёт площади

    Возвращает float - значение площади под графиком при заданных параметрах

def square_my(f, left, right, step):
    Функция подсчёта площали под графиком через формулу трапеций
    f - eval - строка, содержащая функцию
    left - float - левая точка интервала. Точка от которой считается площаль
    right - float - правая точка интервала. Точка на которой заканчивается интервал и подсчёт площади
    step - float - шаг, по которому и считается формула трапеций.

    Возвращает float - значение площади под графиком через формулу трапеций.
   
def expression(i = 0, mtx = None):
    Функция-калькулятор матриц
    Итерация функции происходит без вводных параметров:
    expression()

    На первой итерации вводится матрица и пользователю предлагается произвести различные действия
    Конечная итерация происходит при выборе пользователя 0 - выходе из матрицы

    Возвращает str - строковое сообщение 'Вы вышли из калькулятора'

def matrix_expression(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z):
    Функция-декоратор для матричного выраженния через питоновский eval
    
def matrix_st_calculator(exp, dictionary):
    Функция-калькулятор, которая по заданному выражению высчитывает итоговую матрицу
    exp - выражение, вводимое пользователем
    dictionary - словарь, содержащий все обозначения матриц содержащихся в exp

    Возвращает list - итоговую посчитанную матрицу
   

def letter_to_matrix(st):
    Фукния-ввод, для заданной строки для каждой отдельной буквы английского алфавита присваивает итерируемую матрицу
    st - str - строка, содержащая необходимое для подсчёта выражение

    Возвращаает mtxs - dictionary - словарь, где ключи это обозначения которые встречаются в выражении,
    а значения ключей - сами матрицы

def det_my_matrix(mtx):
    Функция подсчёта определителя матрицы рекурсивным способом

    mtx - list - вводимая матрица как двумерный список

    Возвращает float - значение определителя матрицы
   

def inverse_matrix(mtx):
    Функция, высчитывающая для заданной матрицы обратную матрицу
    и возвращающая строку если таковой нет
    mtx - list - вводимая матрица как двумерный список

    Возвращает
    str - строку 'Матрица вырожденная' в случае если не существует обратной матрицы
    ans - list - обратную матрицу в виде двумерного списка
    
def det_st_matrix(mtx):
    Функция-декоратор метода LA.det библиотеки numpy
   
def csv_reader(string,sep):
    Фукния ввода, позволяющая считать данные из csv-файла

    string - str - строка, содержащая название csv-файла
    sep - str - разделитель, который используется в csv-файле для обозначения сепаратора чисел

def lagranzh(X,Y):
    Фукния, вычисляющая интерполяционный многочлен Лагранжа.

    X - list - писок точек по оси абсцисс интерполируемой фукнции
    Y - list - писок точек по оси ординат интерполируемой фукнции

    Возвращает eval - строку, содержащую вычисленный для заданных данных
    интерполяционный многочлен Лагранжа

def lagranzh_method(m):
    Фунция, которая интерполирует данные методом Лагранжа

    m - ist - двумерный список содердащий точки для интерполирования вида [x,y]

    Возвращает result - list - массив с изначальной точкой + значением, посчитанным методол Лагранжа
    вида [x,y,y1], где y1 - значение, посчитанное методом Лагранжа

def jacobi(arr,x,acc):
    Фукния, реализующая метод простых итераций Якоби.
    arr - list - Исходная СЛАУ
    x - list - список изначальных значений для неизвестных
    acc - float - ограничитель кол-ва знаков после запятой

    Возвращает x - list - массив значений неизвестнных

def jacobian_method(mtx):
    Функция-декоратор функции jacobi
    возвращающая число обусловленности матрицы коэффициентов исходной матрицы
    Выводит прямую матрицу коэффициентов и обратную матрицу коэффициентов исходной матрицы
    Высчитывает и выводит решение СЛАУ методом Якоби
    
    mtx - list - исходная матрица как двумерный список

    Возвращает
    string - str - строку'В матрице притсутсвует строка/столбец состоящий из нулей, не дозволяющий расчёт.' в случае если СЛАУ не считается
    conditional_jac - float - число обусловленности матрицы коэффициентов
    
def GJ_method(mtx1):
    Фукния, реализующая метод решения СЛАУ Гаусса-Жордана
    arr - list - Исходная СЛАУ
    
    Возвращает coeff_vect(mtx) - list - массив значений неизвестнных
    
def jordan_method(mtx):
    Функция-декоратор функции GJ_method
    возвращающая число обусловленности матрицы коэффициентов исходной матрицы
    Выводит прямую матрицу коэффициентов и обратную матрицу коэффициентов исходной матрицы
    Высчитывает и выводит решение СЛАУ методом Гаусса-Жордана
    
    mtx - list - исходная матрица как двумерный список

    Возвращает
    string - str - строку'В матрице притсутсвует строка/столбец состоящий из нулей, не дозволяющий расчёт.' в случае если СЛАУ не считается
    conditional_gauss - float - число обусловленности матрицы коэффициентов

def python_generator():
    Функция-генератор, создающая матрицу с заданными параметрами

def newton_method1(m):
    Функция считающая точки интерполяции методом Ньютона
    
def iteration1():
    Функция-декоратор включающая в себя ввод количества уравнений и вывод их решений
    
def iteration_once():
    Итерация и решение одного уравнения
   
def iteration_system():
    Итерация и решение системы дифференциальных уравнений

def euler(func, x0, y0, z0, a, b, n):
    Метод Эйлера для решения дифференциальных уравнений(Включая задачу Коши)

def euler_once(func, x0, y0, a, b, n):
    Метод Эйлера для одного уравнения(с задачей Коши)
 
def euler_system(func, x0, y0, z0, a, b, n):
    Метод Эйлера для системы уравнений(с задачей Коши)

def eulercauchy(func, x0, y0, z0, a, b, n):
    Решение уравнения или системы дифф.уравнений методом Эйлера-Коши(Включая задачу Коши)

def main():
    Функция итерации матрицы и решения её всеми возможными способами
    Декорирует iteration() - функцию задачи матрицы
    Решает матрицу:
    jacobian_method - методом итераций Якоби.
    jordan_method - методом Жордана-Гаусса

    Возвращает list - решение СЛАУ если нет ошибок
    string - str - сообщение об ошибке, если таковая притсутствует
    main() - рекурсивная итерация в случае предвиденных ошибок
    
   

def iteration():
    Функция итерации матрицы

    Возвращает list - матрицу в виде двумерного списка

def csv_generator():
    Функция создающая матрицу по заданным параметрам и забивающая её в создаваемый csv-файл
    
def transpose_my(mtx):
    Фукнция транспонирующая матрицу

    mtx - list - матрица в виде двумерного списка

    Возвращает trans_mtx - list - транспонированная матрица
   

def transpose_st(mtx):
    Фукнция транспонирующая матрицу с помощью библиотеки python

    mtx - list - матрица в виде двумерного списка

    Возвращает trans_mtx - list - транспонированная матрица"""



import numpy as np
import pandas as pd
import time 
import math
import time
import copy
import csv
import pywt
import webbrowser
import random
from numpy import transpose
from numpy import linalg as LA
from fractions import Fraction
from math import factorial
from sympy import expand
from numpy import inf
from scipy.fft import fft, ifft
from scipy import integrate
from sympy import *
########################
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import rcParams
plt.rcParams['font.size'] = 36
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

A=Symbol('A')
B=Symbol('B')
C=Symbol('C')
D=Symbol('D')
F=Symbol('F')
G=Symbol('G') 
H=Symbol('H')
I=Symbol('I') 
J=Symbol('J') 
K=Symbol('K') 
L=Symbol('L') 
M=Symbol('M') 
N=Symbol('N') 
O=Symbol('O') 
P=Symbol('P') 
Q=Symbol('Q') 
R=Symbol('R') 
S=Symbol('S') 
T=Symbol('T') 
U=Symbol('U') 
V=Symbol('V') 
W=Symbol('W') 
x=Symbol('X') 
y=Symbol('Y') 
z=Symbol('Z')

#Функции, к которым у пользователя нет доступа
#То есть к которым он не сможет обратиться за помощью
#И прочитать документацию

def number(str_input, str_error, str_error2, type_num): #str_input - строка выводимая пользователю при вводе 
                                                        #str_error - ошибка: число не является числом ('строка')
                                                        #str_error2 - число не соответсвует указаным требованиям
                                                        #type_num - все допустимые типы чисел
    print(str_input)
    num = input()
    if 'i' in num:
        num = itojnum(num)
    num.replace(" ", "")
    try:
        check = complex(num) # Проверка: является ли числом (комплексное можем взять от любого числа)
    except ValueError:
        print(str_error)
        return number(str_input, str_error, str_error2, type_num)
    
    if (complex in type_num) and check.imag != 0: # Проверки для комплексных чисел
        return jtoinum(num)
    elif (complex in type_num) and check.imag == 0:
        if (int in type_num):
            if check.real == round(check.real):
                return str(int(check.real))
        if (float in type_num):
            if check.real != round(check.real):
                return str(float(check.real))
        else:
            print(str_error2)
            return number(str_input, str_error, str_error2, type_num)

    elif (float in type_num): # Проверки для вещественных чисел
        if check.imag != 0:
            print(str_error2)
            return number(str_input, str_error, str_error2, type_num)
        if (int in type_num):
            if check.real == round(check.real):
                return str(int(check.real))
        else:
            return str(float(check.real))

    else: # Проверки для целых чисел
        if check.imag != 0:
            print(str_error2)
            return number(str_input, str_error, str_error2, type_num)
        elif check.real != round(check.real):
            print(str_error2)
            return number(str_input, str_error, str_error2, type_num)
        return str(int(check.real))

# Матричный калькулятор

def str_to_complex(mtx):
    for row in mtx:
        for i in range(len(row)):
            row[i]=complex(itojnum(row[i]))
    return(mtx)

def complex_to_num(st):
    if st.imag==0:
        if round(st.real)==st.real:
            return int(st.real)
        else:
            return float(st.real)
    else:
        return complex(st.real, st.imag)
    
# Шаблоны матриц

def mtx1(): # Шаблон 1: Заполнена только главная диагональ.
    try:
        rowcol = list(map(int,input('Введите количество строк(N) и столбцов(M)(N = M): ').split()))
        N = rowcol[0]
        M = rowcol[1]
    except ValueError:
        print('Введено не целое значения строки и/или столбца. Попробуйте ещё раз.')
        return mtx1()
    except IndexError:
        print('Введено Слишком мало значений. Попробуйте ещё раз.')
        return mtx1()
    if N != M:
        print('Матрица должна быть квадратной! Попробуйте ещё раз.')
        return mtx1()
    n = number('Введите число, которое будет на главной диагонали: ', 
               'Введено неверное выражение. Попробуйте ещё раз.',
               'Введен неверный формат числа. Попробуйте ещё раз.',
              [complex, float, int])
    mtx = [['0'] * M for i in range(N)]
    for i in range(N):
        for j in range(M):
            if i == j:
                mtx[i][j] = n 
    return mtx

def mtx2(): # Шаблон 2: Матрица заполняется через арифметическую прогрессию.
    try:
        rowcol = list(map(int,input('Введите количество строк(N) и столбцов(M)(N = M): ').split()))
        N = rowcol[0]
        M = rowcol[1]
    except ValueError:
        print('Введено не целое значения строки и/или столбца. Попробуйте ещё раз.')
        return mtx2()
    except IndexError:
        print('Введено Слишком мало значений. Попробуйте ещё раз.')
        return mtx2()
    if N != M:
        print('Количество столбцов должно быть равно количеству строчек! Попробуйте ещё раз.')
        return mtx2()
    n = number('Введите число, с которого будет начинаться арифметическая прогрессия(целое/комплексное): ',
              'Введено неверное выражение. Попробуйте ещё раз.',
               'Введен неверный формат числа. Попробуйте ещё раз.',
              [complex, int])
    m = int(number('Введите шаг прогрессии(целое число): ',
                  'Введено неверное выражение. Попробуйте ещё раз.',
                  'Введен неверный формат числа. Попробуйте ещё раз.',
                  [int]))
    print(n, m)
    mtx = [['0'] * M for i in range(N)]
    n = itojnum(n)
    comn = complex(n)
    if comn.imag == 0:
        for i in range(N):
            for j in range(M):
                mtx[i][j] = str(abs(i - j) * m + int(n))
    else:
        for i in range(N):
            for j in range(M):
                mtx[i][j] = str(abs(i - j) * m + complex(comn.real, comn.imag + abs(i - j) * m))
    mtx = jtoi(mtx)
    mtx=del_bracket(mtx)
    return mtx

def mtx3(): # Шаблон 3: "Шахматный порядок".
    try:
        rowcol = list(map(int,input('Введите количество строк и столбцов(что бы был рисунок, используйте числа больше 1): ').split()))
        N = rowcol[0]
        M = rowcol[1]
    except ValueError:
        print('Введено не целое значения строки и/или столбца. Попробуйте ещё раз.')
        return mtx3()
    except IndexError:
        print('Введено Слишком мало значений. Попробуйте ещё раз.')
        return mtx3()
    n = number('Введите число, которое будет расставлено по клеткам: ',
                              'Введено неверное выражение. Попробуйте ещё раз.', 
                              'Введено число в неверном формате. Попробуйте ещё раз.',
                              [complex, float, int])
    mtx = [['0'] * M for i in range(N)]
    for i in range(N):
        for j in range(M):
            if (i + j) % 2 == 0:
                mtx[i][j] = n
    return mtx

def mtx4(): # Шаблон 4: Матрица в виде буквы "Е".
    try:
        rowcol = list(map(int,input('Введите количество строк(N) и столбцов(M)(N>=5 и N - нечётное, M>=2): ').split()))
        N = rowcol[0]
        M = rowcol[1]
    except ValueError:
        print('Введено не целое значения строки и/или столбца.')
        return mtx4()
    except IndexError:
        print('Введено слишком мало значений. Попробуйте ещё раз.')
        return mtx4()    
    if N < 5 or  M < 2 or N % 2 == 0:
        print('Не выполнены условия введения матрицы!')
        return mtx4()
    n = number('Введите число, которое будет расставлено по букве E: ',
                              'Введено неверное выражение. Попробуйте ещё раз.', 
                              'Введено число в неверном формате. Попробуйте ещё раз.',
                              [complex, float, int])
    mtx = [['0'] * M for i in range(N)]
    for i in range(M):
        mtx[0][i] = n
        mtx[-1][i] = n
        mtx[N//2][i] = n
    for i in range(N):
        mtx[i][0] = n
    return mtx

def mtx5(): # Шаблон 5: Матрица в виде пирамиды с арифметической прогрессией.
    try:
        rowcol = list(map(int,input('Введите количество строк(N) и столбцов(M)(N = M - оба нечётные): ').split()))
        N = rowcol[0]
        M = rowcol[1]
    except ValueError:
        print('Введено не целое значения строки и/или столбца.')
        return mtx5()
    except IndexError:
        print('Введено Слишком мало значений. Попробуйте ещё раз.')
        return mtx5()
    if N != M or N % 2 == 0:
        print('Не выполнены условия ввода матрицы!')
        return mtx5()
    n = number('Введите число, с которого будет начинаться арифметическая прогрессия(целое/комплексное): ',
              'Введено неверное выражение. Попробуйте ещё раз.',
               'Введен неверный формат числа. Попробуйте ещё раз.',
              [complex, int])
    m = int(number('Введите шаг прогрессии(целое число): ',
                  'Введено неверное выражение. Попробуйте ещё раз.',
                  'Введен неверный формат числа. Попробуйте ещё раз.',
                  [int]))
    mtx = [['0'] * M for i in range(N)]
    n = itojnum(n)
    comn = complex(n)
    if comn.imag == 0:
        for i in range(N):
            for j in range(M):
                ic = abs(i - N//2)
                jc = abs(j - M//2)
                coord = max(ic, jc)
                mtx[i][j] = str(coord * m + int(n))
    else:
        for i in range(N):
            for j in range(M):
                ic = abs(i - N//2)
                jc = abs(j - M//2)
                coord = max(ic, jc)
                mtx[i][j] = str(coord * m + complex(comn.real, comn.imag + coord * m))
    mtx = jtoi(mtx)
    mtx=del_bracket(mtx)
    return mtx

def secret_function():
    webbrowser.open('https://youtu.be/dQw4w9WgXcQ', new=2)
    return [['you have been rick rolled']]

def itoj(mtx):
    ans = []
    for i in range(len(mtx)):
        temp = []
        y = mtx[i]
        for j in y:
            temp.append(j.replace('i','j'))
        ans.append(temp)
    return ans

def jtoi(mtx):
    ans = []
    for i in range(len(mtx)):
        temp = []
        y = mtx[i]
        for j in y:
            temp.append(j.replace('j)','i'))
        ans.append(temp)
    return ans

def del_bracket(mtx):
    ans = []
    for i in range(len(mtx)):
        temp = []
        y = mtx[i]
        for j in y:
            temp.append(j.replace('(',''))
        ans.append(temp)
    return ans

def itojnum(st):
    ans = ''
    for i in str(st):
        ans += i.replace('i','j')
    return ans

def jtoinum(st):
    ans = ''
    for i in str(st):
        ans += i.replace('j','i')
    return ans

# Вычисление матрицы коэффициентов

def coeff_mtx(mtx):
    mtx1 = []
    for i in range(len(mtx)):
        mtx1.append(mtx[i][:-1])
    return mtx1

# Вычисление вектора своюодных членов

def coeff_vect(mtx):
    mtx1 = []
    for i in range(len(mtx)):
        mtx1.append(mtx[i][-1])
    return mtx1

def scales_multiply(mtx1, mtx2):
    N1 = len(mtx1[0])
    M1 = len(mtx2)
    flag = True
    if N1 != M1:
        flag = False
    return flag

def scales_sum(mtx1, mtx2):
    N1 = len(mtx1)
    M1 = len(mtx2)
    N2 = len(mtx1[0])
    M2 = len(mtx2[0])
    flag = True
    if N1 != M1 or N2 != M2:
        flag = False
    return flag

def matrix_multiply_number(mtx):
    num = complex(input('Введите число на которое хотите умножить матрицу: '))
    for row in range(len(mtx)):
        for element in range(len(mtx[0])):
            mtx[row][element] = complex(mtx[row][element]) * num
    return mtx

def matrix_multiply_matrix(mtx1):
    print('Введите матрицу на которую будет произведено умножение.')
    mtx2 = iteration()
    N1 = len(mtx1[0])
    M1 = len(mtx2)
    N2 = len(mtx1)
    M2 = len(mtx2[0])
    flag = scales_multiply(mtx1, mtx2)
    if flag:
        for row in range(len(mtx1)):
            for col in range(len(mtx1[0])):
                mtx1[row][col]=itojnum(str(mtx1[row][col]))
                mtx2[row][col]=itojnum(str(mtx2[row][col]))
        mtx1=str_to_complex(mtx1)
        mtx2=str_to_complex(mtx2)
        mtx = [[0] * M2 for i in range(N2)]
        for row in range(len(mtx1)):
            for col in range(len(mtx1[0])):
                temp = complex(0)
                for num in range(N1):
                    temp += mtx1[row][num] * mtx2[num][col]
                mtx[row][col] = temp
    else:
        print('Введённая матрица не совпадает по размерам для умножения. Ещё раз')
        return matrix_multiply_matrix(mtx1)
    return mtx

def matrix_sum_matrix(mtx1):
    print('Введите матрицу для суммы.')
    mtx2 = iteration()
    flag = scales_sum(mtx1, mtx2)
    N1 = len(mtx1)
    N2 = len(mtx1[0])
    if flag:
        for row in range(len(mtx1)):
            for col in range(len(mtx1[0])):
                mtx1[row][col]=itojnum(str(mtx1[row][col]))
                mtx2[row][col]=itojnum(str(mtx2[row][col]))
        mtx1=str_to_complex(mtx1)
        mtx2=str_to_complex(mtx2)
        mtx = [[0] * N2 for i in range(N1)]
        for i in range(N1):
            for j in range(N2):
                mtx[i][j] = complex(mtx1[i][j]) + complex(mtx2[i][j])
    else:
        print('Введённая матрица не совпадает по размерам для суммы. Ещё раз')
        return matrix_sum_matrix(mtx1)
    return mtx

def matrix_dif_matrix(mtx1):
    print('Введите матрицу для разности.')
    mtx2 = iteration()
    flag = scales_sum(mtx1, mtx2)
    N1 = len(mtx1)
    N2 = len(mtx1[0])
    if flag:
        for row in range(len(mtx1)):
            for col in range(len(mtx1[0])):
                mtx1[row][col]=itojnum(str(mtx1[row][col]))
                mtx2[row][col]=itojnum(str(mtx2[row][col]))
        mtx1=str_to_complex(mtx1)
        mtx2=str_to_complex(mtx2)
        mtx = [[0] * N2 for i in range(N1)]
        for i in range(N1):
            for j in range(N2):
                mtx[i][j] = complex(mtx1[i][j]) - complex(mtx2[i][j])
    else:
        print('Введённая матрица не совпадает по размерам для разности. Ещё раз')
        return matrix_dif_matrix(mtx1)
    return mtx

def default_matrix(): # N - кол-во строк, M - кол-во столбцов
    try:
        rowcol = list(map(int,input('Введите количество строк и столбцов: ').split()))
        N = rowcol[0]
        M = rowcol[1]
        if len(rowcol) > 2:
            print('Введено слишком много значений. Попробуйте ещё раз.')
            return default_matrix()
    except ValueError:
        print('Введено не целое значение строки и/или столбца. Попробуйте ещё раз.')
        return default_matrix()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return default_matrix()
    if N == 0 or M == 0:
        print('Введено нулевое значение! Количество строк и столбцов должно быть минимум 1!!')
        return default_matrix()
    mtx = [[0] * M for i in range(N)]
    for n in range(N):
        for m in range(M):
            mtx[n][m] = number(f'Введите значение для элемента матрицы a[{n + 1}][{m + 1}]: ',
                              'Введено неверное выражение. Попробуйте ещё раз', 
                              'Введено число в неверном формате. Попробуйте ещё раз.',
                              [complex, float, int])
            
    for n in range(len(mtx)):
        #mtx[n].append('|')
        mtx[n].append(number(f'Введите значение для свободного члена {n + 1} строки: ',
                              'Введено неверное выражение. Попробуйте ещё раз',
                              'Введено число в неверном формате. Попробуйте ещё раз.',
                              [complex, float, int]))
    return mtx

def sample_mtx():
    try:
        choice = int(input("Какой вы хотите выбрать шаблон(1,2,3,4,5): "))
        choices_dict = {1: mtx1, 2: mtx2, 3: mtx3, 4: mtx4, 5: mtx5}
        mtx = choices_dict[choice]()
    except KeyError:
        print('Введено неверное значение шаблона. Попробуйте ещё раз.')
        return sample_mtx()
    except ValueError:
        print('Введено неверное значение шаблона. Попробуйте ещё раз.')
        return sample_mtx()
    return mtx

def random_numbers(row,minim,maxi):
    complex_numb=[]
    for i in range(row**3):
        floatnump=random.randint(1,6)
        numb_of_list=random.randint(1,2)
        if numb_of_list==1:
            a=random.randint(minim,maxi)
        else:
            a=round(random.uniform(minim,maxi),floatnump)
        numb_of_list=random.randint(1,2)
        if numb_of_list==1:
            b=random.randint(minim,maxi)
        else:
            b=round(random.uniform(minim,maxi),floatnump)
        complex_numb.append(complex(a,b))
    
    result=[0]*row
    for i in range(row):
        floatnump=random.randint(1,6)
        numb_of_list=random.randint(1,3)
        if numb_of_list==1:
            result[i]=str(random.randint(minim,maxi))
        if numb_of_list==2:
            result[i]=str(round(random.uniform(minim,maxi),floatnump))
        if numb_of_list==3:
            result[i]=str(random.choice(complex_numb))
    
    return result

def newton1_function(X,Y):
    h=X[1]-X[0]
    
    # Найдем конечные разности
    y = copy.deepcopy(Y)
    deltay = [y[0]]
    while len(y) != 1:
        y = [y[i]-y[i-1] for i in range(1,len(y))]
        deltay.append(y[0])

    result=deltay[0]
    
    x = Symbol('x')
    deltax = [eval('x')-X[0]]
    for i in range(1,len(deltay)-1):
        deltax.append(deltax[i-1]*(eval('x') - X[i]))

    for i in range(1,len(deltax)+1):
        deltay[i] /= h**(i) * factorial(i)
        result+=(deltay[i]*deltax[i-1])
    return result

def newton2_function(X,Y):
    h=X[1]-X[0]
    
    # Найдем конечные разности
    y = copy.deepcopy(Y)
    deltay = [y[-1]]
    while len(y) != 1:
        y = [y[i]-y[i-1] for i in range(1,len(y))]
        deltay.append(y[-1])
    
    result=deltay[0]

    x = Symbol('x')
    deltax = [eval('x')-X[-1]]
    for i in range(1,len(deltay)-1):
        deltax.append(deltax[i-1]*(eval('x') - X[len(deltay)-1 - i]))
        
    for i in range(1,len(deltax)+1):
        deltay[i] /= h**(i) * factorial(i)
        result+=(deltay[i]*deltax[i-1])
    return result

def linear_function(X,Y):
    n = len(X)
    sumX2 = sum([X[i] * X[i] for i in range(n)])
    sumXY = sum([X[i] * Y[i] for i in range(n)])
    a = (n * sumXY - sum(X) * sum(Y))/(n * sumX2 - (sum(X))**2)
    b = (sum(Y) - a * sum(X))/n
    x = Symbol('x')
    result = eval('x')*a + b
    return result

def linfuncapprox(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    print('Функция через линейную аппроксимацию: ')
    print(linear_function(X,Y))
    print('Результат вычислений линейной аппроксимации: ')
    x = Symbol('x')
    t = lambdify(x, linear_function(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    print(result)
    result.append(linear_function(X,Y))
    disp = sum([(Y[i] - resnew[i])**2 for i in range(len(resnew))])
    print(f'Величина дисперсии: {disp}')
    result.append(disp)
    return result

def quadratic_function(X,Y):
    n = len(X)
    sumX4 = sum([X[i] * X[i] * X[i] * X[i] for i in range(n)])
    sumX3 = sum([X[i] * X[i] * X[i] for i in range(n)])
    sumX2 = sum([X[i] * X[i] for i in range(n)])
    sumXY = sum([X[i] * Y[i] for i in range(n)])
    sumX2Y = sum([X[i] * X[i] * Y[i] for i in range(n)])
    
    matrix = [[sumX4, sumX3, sumX2, sumX2Y], [sumX3, sumX2, sum(X), sumXY], [sumX2, sum(X), n, sum(Y)]]
    
    GJ_method_abc = GJ_method_2(matrix)
    a = GJ_method_abc[0]
    b = GJ_method_abc[1]
    c = GJ_method_abc[2]
    
    x = Symbol('x')
    result = (eval('x')**2)*a + b*eval('x') + c
    
    return(result)

def quadfuncapprox(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    print('Функция через квадаратичную аппроксимацию: ')
    print(quadratic_function(X,Y))
    print('Результат вычислений через квадратичную аппроксимацию: ')
    x = Symbol('x')
    t = lambdify(x, quadratic_function(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    print(result)
    result.append(quadratic_function(X,Y))
    disp = sum([(Y[i] - resnew[i])**2 for i in range(len(resnew))])
    print(f'Величина дисперсии: {disp}')
    result.append(disp)
    return result

def normdistr_function(X,Y):
    n = len(X)
    
    if not(all([i >= 0 for i in Y]) or all([i <= 0 for i in Y])):
        return 'Чел, это невозможно решить!'
    sign = 1
    if (Y[0] < 0):
        sign = -1
    Y1 = [log(i * sign) for i in Y]
    
    sumX4 = sum([X[i] * X[i] * X[i] * X[i] for i in range(n)])
    sumX3 = sum([X[i] * X[i] * X[i] for i in range(n)])
    sumX2 = sum([X[i] * X[i] for i in range(n)])
    sumX = sum(X)
    
    sumX2Y = sum([X[i] * X[i] * Y1[i] for i in range(n)])
    sumXY = sum([X[i] * Y1[i] for i in range(n)])
    sumY = sum(Y1)
    
    matrix = [[sumX4, sumX3, sumX2,sumX2Y], [sumX3, sumX2, sumX,sumXY], [sumX2, sumX, n, sumY]]
    
    ans = GJ_method_2(matrix)[::-1]
    
    for i in range(len(ans)):
        ans[i] = float(ans[i])

    x = Symbol('x')
    
    c = (- (abs(1 / ans[2]) ** 0.5))
    b = (ans[1] * c ** 2 / 2)
    a = (exp(ans[0] + b ** 2 / c ** 2) * sign)
    
    if b.imag == 0:
        b = b.real
    
    result = a * exp(-((eval('x')-b)**2)/c)
    return result
    
def normdistrapprox(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    print(normdistr_function(X,Y))
    print('Функция нормального распределнеия: ')
    print(normdistr_function(X,Y))
    print('Результат вычислений через аппроксимацию функцией нормального распределения: ')
    x = Symbol('x')
    t = lambdify(x, normdistr_function(X,Y))
    resnew=(t(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    print(result)
    result.append(quadratic_function(X,Y))
    disp = sum([(Y[i] - resnew[i])**2 for i in range(len(resnew))])
    print(f'Величина дисперсии: {disp}')
    result.append(disp)
    return result

def numpy_interpol(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    res = np.interp(X, X, Y)
    result = [[X[i], Y[i], res[i]] for i in range(len(X))]
    return result

def numpy_approx(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    t = np.polyfit(X,Y,6)
    f = np.poly1d(t)
    res = f(X)
    result = [[X[i], Y[i], res[i]] for i in range(len(X))]
    disp = sum([(Y[i] - res[i])**2 for i in range(len(res))])
    result.append(disp)
    return result

def lagranzh_method_forgrp(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    x = Symbol('x')
    a = expand(lagranzh(X,Y))
    t = lambdify(x, lagranzh(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    result.append(a)
    return result

def newton_method1_forgrp(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    x = Symbol('x')
    a = expand(newton1_function(X,Y))
    t = lambdify(x, newton1_function(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    result.append(a)
    return result

def newton_method2_forgrp(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    x = Symbol('x')
    a = expand(newton2_function(X,Y))
    t = lambdify(x, newton2_function(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    result.append(a)
    return result

def linfuncapprox_forgrp(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    x = Symbol('x')
    t = lambdify(x, linear_function(X,Y))
    if type(t(X)) == type(1.0):
        resnew = [t(X) for i in range(len(X))]
    else:
        resnew=(t(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    result.append(linear_function(X,Y))
    disp = sum([(Y[i] - resnew[i])**2 for i in range(len(resnew))])
    result.append(disp)
    return result

def quadfuncapprox_forgrp(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    x = Symbol('x')
    t = lambdify(x, quadratic_function(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    result.append(quadratic_function(X,Y))
    disp = sum([(Y[i] - resnew[i])**2 for i in range(len(resnew))])
    result.append(disp)
    return result

def normdistrapprox_forgrp(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    x = Symbol('x')
    if type(normdistr_function(X,Y)) == type('str'):
        return 'Это очень тяжело рещить для нас('
    t = lambdify(x, normdistr_function(X,Y))
    resnew=(t(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    result.append(normdistr_function(X,Y))
    disp = sum([(Y[i] - resnew[i])**2 for i in range(len(resnew))])
    result.append(disp)
    
# Приводим матрицу к такому виду, что на диагонали стоят еденицы

def diag(mtx1):
    """Функция приводящая матрицу к виду едениц на диагонали

    mtx1 - list - матрица как двумерный список

    Возвращает mtx - list - посчитанную с еденицами на диагонали матрицу"""
    mtx = copy.deepcopy(mtx1)
    for row in range(len(mtx)):
        mtx[row] = [mtx[row][i] / mtx[row][row] for i in range(len(mtx) + 1)]
    return mtx

# Проверка есть ли 0 на диагонали

def is_diag(mtx1):
    """Функция проверки для матрицы если есть нули на главной диагонали

    mtx1 - list - матрица как двумерный список

    Возвращает flag - bool - True или False в зависимости от выполнения условия"""
    mtx = copy.deepcopy(mtx1)
    flag = True
    for row in range(len(mtx)):
        if mtx[row][row] == 0:
            flag = False
    return flag

def is_continue(mtx1):
    """Функция, проверяющая для матрицы необходимость дальнейших подсчётов

    mtx1 - list - матрица как двумерный список

    Возвращает bool - True или False в зависимости от выполнения условия"""
    mtx = copy.deepcopy(mtx1)
    flag = True
    for row in range(len(mtx)):
        col = []
        if mtx[row][:-1] == [0]*len(mtx):
            flag = False
        for j in range(len(mtx)):
            col.append(mtx[j][row])
        if col == [0] * len(mtx):
            flag = False
    return flag

# Приводим матрицу к считаемому виду

def diag_to1(mtx1):
    """Фукния приводящая матрицу к более читаемому для методов решения СЛАУ виду

    mtx1 - list - матрица как двумерный список

    Возвращает diag(mtx) - list - матрицу в виде двумерного списка с более читаемымм видом"""
    mtx = copy.deepcopy(mtx1)
    for i in range(len(mtx)):
        if is_diag(mtx):
            return diag(mtx)
        if mtx[i][i] == 0:
            for j in range(len(mtx)):
                if mtx[j][i] != 0:
                    mtx[i] = [mtx[i][k] + mtx[j][k] for k in range(len(mtx) + 1)]
    return diag(mtx)

#Фукнции, доступные пользователю
#для которых требуется памятка помощи пользователю
# - Документация

def func(x):
    """Функция-декораттор питоновского eval - считывает строку от x и выполняет её"""
    y = eval(input('Введите функцию:')) #Считываем нашу функцию как функцию от переменной
    return y

def diff_my(fx ,x0, dx):
    """Функция вычисления производной в точке x0 методом двусторонней разности
    fx - eval - строка, представляющая собой функцию
    x0 - float - Точка в которой ищется производная методом двусторонней разности
    dx - float - Приращение аргумента по которому считается формула

    Возвращает float - значение производной, посчитаной по формуле"""
    f = lambdify(x, fx)
    y = (f(x0 + dx) - f(x0 - dx)) / (2 * dx) #Ищем производную в точке двусторонней разностью
    return y

def diff_std(f, x, x0):
    """Фунция вычисления производной в точке питоновской библиотекой
    f - eval - строка, представляющая собой функцию
    x - переменная по которой происходит дифференцирование
    x0 - float - Точка в которой ищется производная методом двусторонней разности

    Возвращает float - значение производной, посчитаной по формуле"""
    df = diff(f, x)
    ddx = lambdify(x, df)
    y = ddx(x0)#Ищем производную функцию и затем считаем её в точке(внутренней питоновской библиотекой)
    return y

def inspection(a,b): # Проверка на ОДЗ # inspection(ymydiff,ystdiff)
    """Процедура изменения двух массивов, где значения одного массива принимаются
    за корректные. Массив с некорректными значениями заменяет свои значения на корректные из первого массива
    
    a - list - массив с корректными значениями
    b - list - массив с проверяемыми на некорректность значениями"""
    for num in a:
        if abs(num) > 1000:
             num = np.nan
                
    for num in b:
        if abs(num) > 1000:
            num = np.nan
            
    for i in range(len(b)):
        y = b[i]
        if abs(y) > 1000:
            a[i] = np.nan

    for i in range(len(a)):
        y = a[i]
        st = str(y)
        if 'n' in st:
            a[i] = inf
        

def xrep(x, st): #Функция для замены слишком маленьких значений функции
    """Функция замены зануляемых интерпретатором значений на строки
    x - folat - точка, котороя подставляется в функцию
    st - str - строка функции

    Возвращает строку - выражение, которое не может корректно посчитать интерпретатор
    но наглядное для пользователя"""
    flag = True
    j = 0
    while flag:
        j += 1
        try:
            if st[j] == 'x':
                if st[j - 1] == 'e' and st[j + 1] == 'p':
                    continue
                else:
                    st = st[: j] + '(' + str(x)+ ')' + st[j + 1:]
                    j = 0
        except IndexError:
                break
    return st

def square_integral(f, left, right): # Площадь под графиком через интеграл
    """Фукнция подсчёта площади под функцией через интеграл
    с помощью библиотеки sypmy методом integrate

    f - eval - строка содежащая функцию
    left - float - левая точка интервала. Точка от которой считается площаль
    right - float - правая точка интервала. Точка на которой заканчивается интервал и подсчёт площади

    Возвращает float - значение площади под графиком при заданных параметрах"""
    i = sympy.integrate(f, (x, left, right))
    return i

def square_my(f, left, right, step): #Площадь под графиком через формулу трапеций
    """Функция подсчёта площали под графиком через формулу трапеций
    f - eval - строка, содержащая функцию
    left - float - левая точка интервала. Точка от которой считается площаль
    right - float - правая точка интервала. Точка на которой заканчивается интервал и подсчёт площади
    step - float - шаг, по которому и считается формула трапеций.

    Возвращает float - значение площади под графиком через формулу трапеций."""
    f = lambdify(x, f)
    ans = 0
    pts = np.arange(left, right, step)
    a = pts[0]
    b = pts[-1]
    np.delete(pts, 0)
    np.delete(pts, -1)
    for i in pts:
        ans += f(i)
    ans += f(a) / 2
    ans += f(b) / 2
    ans *= step
    return ans

def expression(i = 0, mtx = None):
    """Функция-калькулятор матриц
    Итерация функции происходит без вводных параметров:
    expression()

    На первой итерации вводится матрица и пользователю предлагается произвести различные действия
    Конечная итерация происходит при выборе пользователя 0 - выходе из матрицы

    Возвращает str - строковое сообщение 'Вы вышли из калькулятора'"""
    if i == 0:
        print('Ввод первой матрицы.')
        mtx = iteration()
    try:
        choice = int(input('Что вы хотите сделать с этой матрицей: \n 1 - Умножить на матрицу \n 2 - Умножить на число \n 3 - Сложить с матрицей \n 4 - Вычесть из неё матрицу \n 0 - Выход из калькулятора. \n'))
        if choice == 0:
            print('Успешное завершение программы.😺')
            return 'Вы вышли из калькулятора'
        choices_dict = {1: matrix_multiply_matrix, 2: matrix_multiply_number , 3: matrix_sum_matrix, 4: matrix_dif_matrix}
        mtx1 = choices_dict[int(choice)](mtx)
    except KeyError:
        print('Введено неверное значение ввода калькулятора. Попробуйте ещё раз. 1')
        return expression()
    except ValueError:
        print('Введено неверное значение ввода калькулятора. Попробуйте ещё раз. 2')
        return expression()
    print('Полученная матрица: ')
    mtx2=[[0]*len(mtx1) for i in range(len(mtx1))]
    for i in range(len(mtx1)):
        for j in range(len(mtx1[0])):
            mtx2[i][j]=complex_to_num(mtx1[i][j])
    #print(mtx2)
    for i in range(len(mtx1)):
        for j in range(len(mtx1[0])):
            mtx2[i][j]=str(mtx2[i][j])
    mtx2=jtoi(mtx2)
    mtx2=del_bracket(mtx2)
    #print(mtx2)
    for row in mtx2:
        for num in row:
            print('{:12s}'.format(num), end = ' ')
        print('')
    return expression(i + 1, mtx1)

# Матричный калькулятор через библиотеки numpy и sympy

def matrix_expression(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z):
    """Функция-декоратор для матричного выраженния через питоновский eval"""
    y = eval(input('Введите выражение для матричного калькулятора: '))
    return y

def matrix_st_calculator(exp, dictionary):
    """Функция-калькулятор, которая по заданному выражению высчитывает итоговую матрицу
    exp - выражение, вводимое пользователем
    dictionary - словарь, содержащий все обозначения матриц содержащихся в exp

    Возвращает list - итоговую посчитанную матрицу"""
    alph = list(dictionary.keys())
    f = lambdify(alph, exp)
    all_matrix=[]
    for key, val in dictionary.items():
        mtx=np.array(str_to_complex(val))
        all_matrix.append(mtx)
    y=f(*all_matrix)
    return y

def letter_to_matrix(st):
    """Фукния-ввод, для заданной строки для каждой отдельной буквы английского алфавита присваивает итерируемую матрицу
    st - str - строка, содержащая необходимое для подсчёта выражение

    Возвращаает mtxs - dictionary - словарь, где ключи это обозначения которые встречаются в выражении,
    а значения ключей - сами матрицы"""
    alph = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '1234567890'
    mtxs = {}
    for i in st:
        if i in alph:
            try:
                check = mtxs[i]
            except KeyError:
                print(f'Введите значение для матрицы {i}: ')
                mtxs[i] = iteration()
    return mtxs

def det_my_matrix(mtx):
    """Функция подсчёта определителя матрицы рекурсивным способом

    mtx - list - вводимая матрица как двумерный список

    Возвращает float - значение определителя матрицы"""
    Lmtx=len(mtx)
    
    if Lmtx==1:
        return mtx[0][0]
    if Lmtx==2:
        return mtx[0][0]*mtx[1][1]-(mtx[0][1]*mtx[1][0])
    
    result=0
    for i in range(Lmtx):
        
        factor=1
        if i % 2:
            factor=-1
            
        mtx2=[]
        for row in range(Lmtx):
            mtx3=[]
            for col in range(Lmtx):
                if row!=0 and col!=i:
                    mtx3.append(mtx[row][col])
            if mtx3:
                mtx2.append(mtx3)
        
        result+=factor*mtx[0][i]*det_my_matrix(mtx2)
    return(result)

def inverse_matrix(mtx):
    """Функция, высчитывающая для заданной матрицы обратную матрицу
    и возвращающая строку если таковой нет
    mtx - list - вводимая матрица как двумерный список

    Возвращает
    str - строку 'Матрица вырожденная' в случае если не существует обратной матрицы
    ans - list - обратную матрицу в виде двумерного списка"""
    Lmtx = len(mtx)
    mult = det_my_matrix(mtx)
    if mult == 0:
        return 'Матрица вырожденная'
    ans = [[0] * Lmtx for i in range(Lmtx)]
    for i in range(Lmtx):  
        for j in range(Lmtx):
            factor=1
            if (i+j) % 2:
                factor=-1
            mtx2 = []
            for i1 in range(Lmtx):
                if i1 != i:
                    mtx3 = []
                    for j1 in range(Lmtx):
                        if j1 != j:
                            mtx3.append(mtx[i1][j1])
                    mtx2.append(mtx3)
            ans[j][i] = factor * det_my_matrix(mtx2) / mult
    return ans

# Решение через библиотеку numpy

def det_st_matrix(mtx):
    """Функция-декоратор метода LA.det библиотеки numpy"""
    return LA.det(np.array(mtx))

def csv_reader(string,sep):
    """Фукния ввода, позволяющая считать данные из csv-файла

    string - str - строка, содержащая название csv-файла
    sep - str - разделитель, который используется в csv-файле для обозначения сепаратора чисел"""
    with open(string,newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=sep)
        lst=[]
        for row in reader:
            lst.append(list(row))
    return lst

#Интерполяция методом Лагранжа

def lagranzh(X,Y):
    """Фукния, вычисляющая интерполяционный многочлен Лагранжа.

    X - list - писок точек по оси абсцисс интерполируемой фукнции
    Y - list - писок точек по оси ординат интерполируемой фукнции

    Возвращает eval - строку, содержащую вычисленный для заданных данных
    интерполяционный многочлен Лагранжа"""
    numerator = [1]*len(X)
    x = Symbol('x')
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                numerator[i] *= eval('x')-X[j]
        numerator[i] *= Y[i]
    
    denominator = [1]*len(X)
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                denominator[i] *= X[i]-X[j]
    
    result = 0
    for i in range(len(numerator)):
        result += numerator[i]/denominator[i]
        
    return result

def lagranzh_method(m):
    """Фунция, которая интерполирует данные методом Лагранжа

    m - ist - двумерный список содердащий точки для интерполирования вида [x,y]

    Возвращает result - list - массив с изначальной точкой + значением, посчитанным методол Лагранжа
    вида [x,y,y1], где y1 - значение, посчитанное методом Лагранжа"""
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    print('Интерполяционный многочлен Лагранжа: ')
    a = expand(lagranzh(X,Y))
    print(a)
    print('Результат вычислений методом Лагранжа: ')
    x = Symbol('x')
    t = lambdify(x, lagranzh(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    print(result)
    result.append(a)
    return result

# Вычисление матодом простых итераций Якоби

def jacobi(arr,x,acc):
    """Фукния, реализующая метод простых итераций Якоби.
    arr - list - Исходная СЛАУ
    x - list - список изначальных значений для неизвестных
    acc - float - ограничитель кол-ва знаков после запятой

    Возвращает x - list - массив значений неизвестнных"""
    arr1 = coeff_mtx(arr)
    vect = coeff_vect(arr)
    D = np.diag(arr1)
    R = arr1 - np.diagflat(D)
    x1 = [i for i in x]
    x = (vect - np.dot(R,x)) / D
    fin = abs(x1 - x)
    itr = 0
    while max(fin)>=acc:
        if itr >= 100:
            return 'Матрица расходится'
        itr += 1
        x1 = [i for i in x]
        x = (vect - np.dot(R,x)) / D
        fin = abs(x1 - x)
    return x

# Метод простых итераций Якоби

def jacobian_method(mtx):
    """Функция-декоратор функции jacobi
    возвращающая число обусловленности матрицы коэффициентов исходной матрицы
    Выводит прямую матрицу коэффициентов и обратную матрицу коэффициентов исходной матрицы
    Высчитывает и выводит решение СЛАУ методом Якоби
    
    mtx - list - исходная матрица как двумерный список

    Возвращает
    string - str - строку'В матрице притсутсвует строка/столбец состоящий из нулей, не дозволяющий расчёт.' в случае если СЛАУ не считается
    conditional_jac - float - число обусловленности матрицы коэффициентов"""
    mtx1 = str_to_complex(mtx)
    coeff = coeff_mtx(mtx1)
    vect = coeff_vect(mtx1)
    need = is_continue(mtx1)
    if need == False:
        return 'В матрице притсутсвует строка/столбец состоящий из нулей, не дозволяющий расчёт.'
    mtx1 = diag_to1(mtx1)
    n = len(mtx)
    print('Прямая матрица коэффициентов:')
    for i in range(n):
        print(coeff[i])
    rev = inverse_matrix(coeff)
    print('Обратная матрица коэффициентов:')
    for i in range(n):
        print(rev[i])
    print('Решение СЛАУ методом простых итераций Якоби:')
    mtx2 = np.array(mtx1)
    x = np.array([0 for i in range(n)])
    acc = 0.001
    sol = jacobi(mtx2, x, acc)
    print(sol)
    print('Число обусловленности Матрицы Коэффициентов A: ')
    conditional_jac = LA.cond(coeff)
    print(conditional_jac)
    return conditional_jac

#Вычилсение методом Гаусаа-Жордана

def GJ_method(mtx1):
    """Фукния, реализующая метод решения СЛАУ Гаусса-Жордана
    arr - list - Исходная СЛАУ
    
    Возвращает coeff_vect(mtx) - list - массив значений неизвестнных"""
    mtx = copy.deepcopy(mtx1)
    n = len(mtx)
    if det_my_matrix(mtx) == 0:
        return 'Вырожденная матрица. Нормально не считается этим методом'
    for itr in range(n):
        mtx[itr] = [mtx[itr][i] / mtx[itr][itr] for i in range(n + 1)]
        for col in range(n):
            if col != itr:
                mtx[col] = [mtx[col][i] - mtx[itr][i] * mtx[col][itr] for i in range(n + 1)]
    for row in mtx:
        for i in range(len(row)):
            row[i] = complex_to_num(row[i])
            if abs(row[i]) < 10 ** -10:
                row[i] = 0
    return coeff_vect(mtx)

#Вычилсение методом Гаусаа-Жордана

def GJ_method_2(mtx1):
    mtx = copy.deepcopy(mtx1)
    n = len(mtx)
    if det_my_matrix(mtx) == 0:
        return 'Вырожденная матрица. Нормально не считается этим методом'
    for itr in range(n):
        mtx[itr] = [mtx[itr][i] / mtx[itr][itr] for i in range(n + 1)]
        for col in range(n):
            if col != itr:
                mtx[col] = [mtx[col][i] - mtx[itr][i] * mtx[col][itr] for i in range(n + 1)]
    return coeff_vect(mtx)

# Метод Гаусаа-Жордана  
    
def jordan_method(mtx):
    """Функция-декоратор функции GJ_method
    возвращающая число обусловленности матрицы коэффициентов исходной матрицы
    Выводит прямую матрицу коэффициентов и обратную матрицу коэффициентов исходной матрицы
    Высчитывает и выводит решение СЛАУ методом Гаусса-Жордана
    
    mtx - list - исходная матрица как двумерный список

    Возвращает
    string - str - строку'В матрице притсутсвует строка/столбец состоящий из нулей, не дозволяющий расчёт.' в случае если СЛАУ не считается
    conditional_gauss - float - число обусловленности матрицы коэффициентов"""
    mtx1 = str_to_complex(mtx)
    coeff = coeff_mtx(mtx1)
    vect = coeff_vect(mtx1)
    need = is_continue(mtx1)
    if need == False:
        return 'В матрице притсутсвует строка/столбец состоящий из нулей, не дозволяющий расчёт.'
    mtx1 = diag_to1(mtx1)
    n = len(mtx)
    print('Прямая матрица коэффициентов:')
    for i in range(n):
        print(coeff[i])
    rev = inverse_matrix(coeff)
    print('Обратная матрица коэффициентов:')
    for i in range(n):
        print(rev[i])
    print('Решение СЛАУ методом Жордана-Гаусса:')
    sol = GJ_method(mtx1)
    print(sol)
    print('Число обусловленности Матрицы Коэффициентов A: ')
    conditional_gauss = LA.cond(coeff)
    print(conditional_gauss)
    return conditional_gauss

# Метод Гаусаа-Жордана для правильных дробей
    
def jordan_method_2(mtx):
    mtx1 = numbers_to_fractions(mtx)
    coeff = coeff_mtx(mtx1)
    vect = coeff_vect(mtx1)
    need = is_continue(mtx1)
    if need == False:
        return 'В матрице притсутсвует строка/столбец состоящий из нулей, не дозволяющий расчёт.'
    mtx1 = diag_to1(mtx1)
    n = len(mtx)
    print('Прямая матрица коэффициентов:')
    for i in range(n):
        print(coeff[i])
    rev = inverse_matrix(coeff)
    print('Обратная матрица коэффициентов:')
    for i in range(n):
        print(rev[i])
    print('Решение СЛАУ методом Жордана-Гаусса для Дробей:')
    sol = GJ_method_2(mtx1)
    for i in range(len(sol)):
        print(f'Значение x[{i + 1}] = {sol[i]}')
    for i in range(len(coeff)):
        for j in range(len(coeff[i])):
            coeff[i][j] = float(coeff[i][j])
    conditional_gauss = LA.cond(coeff)
    print('Число обусловленности Матрицы Коэффициентов A: ')
    print(conditional_gauss)
    return conditional_gauss

def python_generator():
    """Функция-генератор, создающая матрицу с заданными параметрами"""
    try:
        rowcol = list(map(int,input('Введите количество строк и столбцов (N M): ').split()))
        N = rowcol[0]
        M = rowcol[1]
        if len(rowcol) > 2:
            print('Введено слишком много значений. Попробуйте ещё раз.')
            return python_generator()
    except ValueError:
        print('Введено не целое значение строки и/или столбца. Попробуйте ещё раз.')
        return python_generator()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return python_generator()
    if N == 0 or M == 0:
        print('Введено нулевое значение! Количество строк и столбцов должно быть минимум 1!!')
        return python_generator()

    try:
        minmax = list(map(int,input('Введите минимальное и максимальное значене для элемента матрицы (также для мнимой части комплексного числа) (min max): ').split()))
        mini = minmax[0]
        maxi = minmax[1]
    except ValueError:
        print('Ошибка ввода. Попробуйте ещё раз.')
        return python_generator()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return python_generator()
    if mini > maxi:
        print(f'Минимальное число не может быть больше максимального ({mini}!>{maxi})!!')
        return python_generator()
    
    result=[]
    for i in range(M):
        result.append(random_numbers(N,mini,maxi))
    
    for row in range(len(result)):
        #result[row].append('|')
        result[row].append(random_numbers(1,mini,maxi))
        result[row][-1]=str(result[row][-1][0])
        
    result=jtoi(result)
    result=del_bracket(result)
        
    return result

def newton_method1(m):
    """Функция считающая точки интерполяции методом Ньютона"""
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    print('Первая интерполяционная формула Ньютона: ')
    a = expand(newton1_function(X,Y))
    print(a)
    print('Результат вычислений методом Ньютона 1): ')
    x = Symbol('x')
    t = lambdify(x, newton1_function(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    print(result)
    result.append(a)
    return result

def newton_method2(m):
    X = np.array([m[i][0] for i in range(len(m))])
    Y = np.array([m[i][1] for i in range(len(m))])
    print('Вторая интерполяционная формула Ньютона: ')
    a = expand(newton2_function(X,Y))
    print(a)
    print('Результат вычислений методом Ньютона 2): ')
    x = Symbol('x')
    t = lambdify(x, newton2_function(X,Y))
    resnew=t(np.array(X)).tolist()
    result = [[X[i],Y[i],resnew[i]] for i in range(len(resnew))]
    print(result)
    result.append(a)
    return result


def iteration1():
    """Функция-декоратор включающая в себя ввод количества уравнений и вывод их решений"""
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    
    system = input('Хотите ли вы ввести систему из двух уравнений? (Да/Нет):')
    
    if system.lower() == 'нет':
        func = function(x,y,z,1)
        z0 = '-'
        try:
            x0y0 = list(map(float,input('Введите начальные условия [x0,y0]: ').split()))
            x0 = x0y0[0]
            y0 = x0y0[1]
        except ValueError:
            print('Ошибка ввода. Попробуйте ещё раз.')
            return iteration1()
        except IndexError:
            print('Введено слишком мало чисел. Попробуйте ещё раз.')
            return iteration1()
        
        try:
            ab = list(map(int,input('Введите желаемый интервал [a,b]: ').split()))
            a = ab[0]
            b = ab[1]
        except ValueError:
            print('Ошибка ввода. Попробуйте ещё раз.')
            return iteration1()
        except IndexError:
            print('Введено слишком мало чисел. Попробуйте ещё раз.')
            return iteration1()
        
        if a > b:
            print(f'Ошибка в вводе интервала! ({a}!>{b})!!')
            return iteration1()
    
    elif system.lower() == 'да':
        func = function(x,y,z,2)
        
        try:
            x0y0 = list(map(float,input('Введите начальные условия [x0,y0,z0]: ').split()))
            x0 = x0y0[0]
            y0 = x0y0[1]
            z0 = x0y0[2]
        except ValueError:
            print('Ошибка ввода. Попробуйте ещё раз.')
            return iteration1()
        except IndexError:
            print('Введено слишком мало чисел. Попробуйте ещё раз.')
            return iteration1()
    
        try:
            ab = list(map(int,input('Введите желаемый интервал [a,b]: ').split()))
            a = ab[0]
            b = ab[1]
        except ValueError:
            print('Ошибка ввода. Попробуйте ещё раз.')
            return iteration1()
        except IndexError:
            print('Введено слишком мало чисел. Попробуйте ещё раз.')
            return iteration1()
        if a > b:
            print(f'Ошибка в вводе интервала! ({a}!>{b})!!')
            return iteration1()
        
    n = int(input('Введите количество точек (n): '))
    
    return(func,x0,y0,z0,a,b,n)

# Одно уравнение

def iteration_once():
    """Итерация и решение одного уравнения"""
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    
    func = function(x,y,z,1)
    try:
        x0y0 = list(map(float,input('Введите начальные условия [x0,y0]: ').split()))
        x0 = x0y0[0]
        y0 = x0y0[1]
    except ValueError:
        print('Ошибка ввода. Попробуйте ещё раз.')
        return iteration_once()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return iteration_once()

    try:
        ab = list(map(int,input('Введите желаемый интервал [a,b]: ').split()))
        a = ab[0]
        b = ab[1]
    except ValueError:
        print('Ошибка ввода. Попробуйте ещё раз.')
        return iteration_once()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return iteration_once()

    if a > b:
        print(f'Ошибка в вводе интервала! ({a}!>{b})!!')
        return iiteration_once()
        
    n = int(input('Введите количество точек (n): '))
    
    return(func,x0,y0,a,b,n)

# Система уравнений

def iteration_system():
    """Итерация и решение системы дифференциальных уравнений"""
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    func = function(x,y,z,2)

    try:
        x0y0 = list(map(float,input('Введите начальные условия [x0,y0,z0]: ').split()))
        x0 = x0y0[0]
        y0 = x0y0[1]
        z0 = x0y0[2]
    except ValueError:
        print('Ошибка ввода. Попробуйте ещё раз.')
        return iteration_system()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return iteration_system()
    
    try:
        ab = list(map(int,input('Введите желаемый интервал [a,b]: ').split()))
        a = ab[0]
        b = ab[1]
    except ValueError:
        print('Ошибка ввода. Попробуйте ещё раз.')
        return iteration_system()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return iteration_system()
    if a > b:
        print(f'Ошибка в вводе интервала! ({a}!>{b})!!')
        return iteration_system()
        
    n = int(input('Введите количество точек (n): '))
    
    return(func,x0,y0,z0,a,b,n)
    return result

# Два в одном

def euler(func, x0, y0, z0, a, b, n):
    """Метод Эйлера для решения дифференциальных уравнений(Включая задачу Коши)"""
    h = (b-a)/n
    x = np.arange(x0,x0+(b-a),h)
    X = Symbol('x')
    Y = Symbol('y')
    Z = Symbol('z')
    if type(z0) == type(''):
        res = [[i, x[i], 0] for i in range(n)]
        res[0][2] = y0
        for i in range(1,n):
            res[i][0] = i
            res[i][1] = x[i]
            res[i][2] = res[i-1][2] + (h*func[0].subs([(X,res[i-1][1]),(Y,res[i-1][2])]))
    
    else:
        res = [[i, x[i], 0, 0] for i in range(n)]
        res[0][2] = y0
        res[0][3] = z0
        for i in range(1,n):
            res[i][0] = i
            res[i][1] = x[i]
            res[i][2] = res[i-1][2] + (h*func[0].subs([(X,res[i-1][1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))
            res[i][3] = res[i-1][3] + (h*func[1].subs([(X,res[i-1][1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))
    return res

# Одно уравнение

def euler_once(func, x0, y0, a, b, n):
    """Метод Эйлера для одного уравнения(с задачей Коши)"""
    h = (b-a)/n
    x = np.arange(x0,x0+(b-a),h)
    X = Symbol('x')
    Y = Symbol('y')
    Z = Symbol('z')
    res = [[i, x[i], 0] for i in range(n)]
    res[0][2] = y0
    for i in range(1,n):
        res[i][0] = i
        res[i][1] = x[i]
        res[i][2] = res[i-1][2] + (h*func[0].subs([(X,res[i-1][1]),(Y,res[i-1][2])]))
    return res

# Система уравнений

def euler_system(func, x0, y0, z0, a, b, n):
    """Метод Эйлера для системы уравнений(с задачей Коши)"""
    h = (b-a)/n
    x = np.arange(x0,x0+(b-a),h)
    X = Symbol('x')
    Y = Symbol('y')
    Z = Symbol('z')
    res = [[i, x[i], 0, 0] for i in range(n)]
    res[0][2] = y0
    res[0][3] = z0
    for i in range(1,n):
        res[i][0] = i
        res[i][1] = x[i]
        res[i][2] = res[i-1][2] + (h*func[0].subs([(X,res[i-1][1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))
        res[i][3] = res[i-1][3] + (h*func[1].subs([(X,res[i-1][1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))
    return res

# Два в одном

def eulercauchy(func, x0, y0, z0, a, b, n):
    """Решение уравнения или системы дифф.уравнений методом Эйлера-Коши(Включая задачу Коши)"""
    h = (b-a)/n
    x = np.arange(x0,x0+(b-a),h)
    X = Symbol('x')
    Y = Symbol('y')
    Z = Symbol('z')
    if type(z0) == type(''):
        _y = [0]*n
        _y[0] = y0
        for i in range(1,n):
            _y[i] = _y[i-1] + (h*func[0].subs([(X,x[i-1]),(Y,_y[i-1])]))
        res = [[i, x[i], 0] for i in range(n)]
        res[0][2] = y0
        for i in range(1,n):
            res[i][2] = res[i-1][2] + (h/2) * ((func[0].subs([(X,x[i-1]),(Y,res[i-1][2])]))+(func[0].subs([(X,x[i]),(Y,_y[i])])))
    
    else:
        _y = [0]*n
        _z = [0]*n
        _y[0] = y0
        _z[0] = z0
        for i in range(1,n):
            _y[i] = _y[i-1] + (h*func[0].subs([(X,x[i-1]),(Y,_y[i-1]),(Z,_z[i-1])]))
            _z[i] = _z[i-1] + (h*func[1].subs([(X,x[i-1]),(Y,_y[i-1]),(Z,_z[i-1])]))
        res = [[i, x[i], 0, 0] for i in range(n)]
        res[0][2] = y0
        res[0][3] = z0
        for i in range(1,n):
            res[i][2] = res[i-1][2] + (h/2) * ((func[0].subs([(X,x[i-1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))+(func[0].subs([(X,x[i]),(Y,_y[i]),(Z,_z[i])])))
            res[i][3] = res[i-1][3] + (h/2) * ((func[1].subs([(X,x[i-1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))+(func[1].subs([(X,x[i]),(Y,_y[i]),(Z,_z[i])])))
    return res

# Одно уравнение

def eulercauchy_once(func, x0, y0, a, b, n):
    h = (b-a)/n
    x = np.arange(x0,x0+(b-a),h)
    X = Symbol('x')
    Y = Symbol('y')
    Z = Symbol('z')
    _y = [0]*n
    _y[0] = y0
    for i in range(1,n):
        _y[i] = _y[i-1] + (h*func[0].subs([(X,x[i-1]),(Y,_y[i-1])]))
    res = [[i, x[i], 0] for i in range(n)]
    res[0][2] = y0
    for i in range(1,n):
        res[i][2] = res[i-1][2] + (h/2) * ((func[0].subs([(X,x[i-1]),(Y,res[i-1][2])]))+(func[0].subs([(X,x[i]),(Y,_y[i])])))
    return res

# Система уравнений

def eulercauchy_system(func, x0, y0, z0, a, b, n):
    h = (b-a)/n
    x = np.arange(x0,x0+(b-a),h)
    X = Symbol('x')
    Y = Symbol('y')
    Z = Symbol('z')
    _y = [0]*n
    _z = [0]*n
    _y[0] = y0
    _z[0] = z0
    for i in range(1,n):
        _y[i] = _y[i-1] + (h*func[0].subs([(X,x[i-1]),(Y,_y[i-1]),(Z,_z[i-1])]))
        _z[i] = _z[i-1] + (h*func[1].subs([(X,x[i-1]),(Y,_y[i-1]),(Z,_z[i-1])]))
    res = [[i, x[i], 0, 0] for i in range(n)]
    res[0][2] = y0
    res[0][3] = z0
    for i in range(1,n):
        res[i][2] = res[i-1][2] + (h/2) * ((func[0].subs([(X,x[i-1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))+(func[0].subs([(X,x[i]),(Y,_y[i]),(Z,_z[i])])))
        res[i][3] = res[i-1][3] + (h/2) * ((func[1].subs([(X,x[i-1]),(Y,res[i-1][2]),(Z,res[i-1][3])]))+(func[1].subs([(X,x[i]),(Y,_y[i]),(Z,_z[i])])))
    return res

def odeint_scp(func, x0, y0, a, b, n):
    h = (b-a)/n
    x = np.arange(x0,x0+(b-a),h)
    X = Symbol('x')
    Y = Symbol('y')
    Z = Symbol('z')
    f = lambdify([X,Y],func[0])
    res = [[i, x[i], 0] for i in range(n)]
    y = odeint(f,y0,np.array(x))
    for i in range(n):
        res[i][2] = float(y[i])
    return res

def wav_sinus_coeff(coeffs):
    a4, d4, d3, d2, d1 = coeffs
    result = sqrt(np.std(d4) + np.std(d3) + np.std(d2) + np.std(d1)) / np.std(a4)
    return result

def diff_my1(y1, y0, dx):
    y = (y1 - y0) / dx 
    return y

def function(x,y,z,col):
    if col == 1:
        f = [eval(input("y' = "))]
    elif col == 2:
        f1 = eval(input("y' = "))
        f2 = eval(input("z' = "))
        f = [f1,f2]
    return f

def numbers_to_fractions(mtx):
    print(mtx)
    mtx1 = []
    for row in range(len(mtx)):
        mtx2 = []
        for col in range(len(mtx[row])):
            if 'i' in str(mtx[row][col]):
                return 'Функция не работает с комплексными числами'
            mtx2.append(Fraction(complex_to_num(mtx[row][col])))
        mtx1.append(mtx2)
    print(mtx1)
    return mtx1

# Основное тело программы

def main():
    """Функция итерации матрицы и решения её всеми возможными способами
    Декорирует iteration() - функцию задачи матрицы
    Решает матрицу:
    jacobian_method - методом итераций Якоби.
    jordan_method - методом Жордана-Гаусса

    Возвращает list - решение СЛАУ если нет ошибок
    string - str - сообщение об ошибке, если таковая притсутствует
    main() - рекурсивная итерация в случае предвиденных ошибок
    """
    matrix=iteration()
    print('Введённая матрица:')
    print(matrix)
    print('Матрица коэфициентов:(Будет выводиться М а если матрица вырожденная)')
    print(coeff_mtx(matrix))
    print('Вектор свободных значений:')
    print(coeff_vect(matrix))
    jac = jacobian_method(matrix)
    if type(jac) == type('питса'):
        return jac
    if jac > 100:
        return 'Программа завершилась.'
    gauss = jordan_method(matrix)
    if gauss > 100:
        return 'Программа завершилась.'
    for row in matrix:
        for col in row:
            if 'i' in str(col):
                print('Нельзя считать дроби с комплексными числами.')
                print('Хотите ли попробовать ввести матрицу ещё раз? \n 1 - да \n 2 - нет')
                try:    
                    choice = int(input())
                    ext = lambda: 'Программа завершилась.'
                    choices_dict = {1: main, 2: ext}
                    mtx = choices_dict[choice]()
                except KeyError:
                    print('Введено неверное значение для ответа на вопрос. Запущен повторный ввод матрицы')
                    return main()
                except ValueError:
                    print('Введено неверное значение для ответа на вопрос. Запущен повторный ввод матрицы')
                    return main()
                return mtx
    try:
        gauss2 = jordan_method_2(matrix)
    except np.linalg.LinAlgError as err:
        print('Ошибка вычислений. Введена вырожденная матрица для которой не считается число обусловленности')
        print('Хотите ли попробовать ввести матрицу ещё раз? \n 1 - да \n 2 - нет')
        try:    
            choice = int(input())
            ext = lambda: 'Программа завершилась.'
            choices_dict = {1: main, 2: ext}
            mtx = choices_dict[choice]()
        except KeyError:
            print('Введено неверное значение для ответа на вопрос. Запущен повторный ввод матрицы')
            return main()
        except ValueError:
            print('Введено неверное значение для ответа на вопрос. Запущен повторный ввод матрицы')
            return main()
        return mtx

def iteration():
    """Функция итерации матрицы

    Возвращает list - матрицу в виде двумерного списка"""
    print("Как вы хотите ввести матрицу:\n 1 - С кливаитуры\n 2 - Рандомная генерация в python\n 3 - CSV Файл")
    try:    
        choice = int(input('Вы ввели: '))
        choices_dict = {1: default_matrix, 2: python_generator , 3: csv_generator, 25102009: secret_function}
        mtx = choices_dict[choice]()
    except KeyError:
        print('Введено неверное значение ввода матрицы. Попробуйте ещё раз.')
        return iteration()
    except ValueError:
        print('Введено неверное значение ввода матрицы. Попробуйте ещё раз.')
        return iteration()
    return mtx

def csv_generator():
    """Функция создающая матрицу по заданным параметрам и забивающая её в создаваемый csv-файл"""
    try:
        rowcol = list(map(int,input('Введите количество строк и столбцов (N M): ').split()))
        N = rowcol[0]
        M = rowcol[1]
        if len(rowcol) > 2:
            print('Введено слишком много значений. Попробуйте ещё раз.')
            return csv_generator()
    except ValueError:
        print('Введено не целое значение строки и/или столбца. Попробуйте ещё раз.')
        return csv_generator()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return csv_generator()
    if N == 0 or M == 0:
        print('Введено нулевое значение! Количество строк и столбцов должно быть минимум 1!!')
        return csv_generator()

    try:
        minmax = list(map(int,input('Введите минимальное и максимальное значене для элемента матрицы (также для мнимой части комплексного числа) (min max): ').split()))
        mini = minmax[0]
        maxi = minmax[1]
    except ValueError:
        print('Ошибка ввода. Попробуйте ещё раз.')
        return csv_generator()
    except IndexError:
        print('Введено слишком мало чисел. Попробуйте ещё раз.')
        return csv_generator()
    if mini > maxi:
        print(f'Минимальное число не может быть больше максимального ({mini}!>{maxi})!!')
        return csv_generator()
    
    result=[]
    for i in range(M):
        result.append(random_numbers(N,mini,maxi))
        
    result=jtoi(result)
    result=del_bracket(result)
    
    with open('MatrixCalculatorDurdKudr.csv','w',newline='') as csvfile:
        writer=csv.writer(csvfile,delimiter=';')
        for row in result:
            writer.writerow(row)

    Matrix_in=[]
    with open('MatrixCalculatorDurdKudr.csv',newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=';')
        Matrix_in=[]
        for row in reader:
            Matrix_in.append(list(row))
    return Matrix_in

# Транспонирование матрицы
def transpose_my(mtx):
    """Фукнция транспонирующая матрицу

    mtx - list - матрица в виде двумерного списка

    Возвращает trans_mtx - list - транспонированная матрица"""
    N = len(mtx)
    M = len(mtx[0])
    trans_mtx = [[0 for j in range(N)] for i in range(M)]
    for i in range(N):
        for j in range(M):
            trans_mtx[j][i] = mtx[i][j]
    return trans_mtx

def transpose_st(mtx):
    """Фукнция транспонирующая матрицу с помощью библиотеки python

    mtx - list - матрица в виде двумерного списка

    Возвращает trans_mtx - list - транспонированная матрица"""
    trans_mtx=np.array(mtx)
    trans_mtx=trans_mtx.transpose()
    return trans_mtx
