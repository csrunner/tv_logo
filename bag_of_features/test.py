# -*- coding:utf-8 -*-
__author__ = 'shichao'


def Func1(par):
    print("Func1")

    print(par)


def Func2(par):
    print("Func2")

    print(par)


@Func1
@Func2
def Func3():
    print("Func3")

    return 9

# def main():
#     Func3()
#
# if __name__ == '__main__':
#     main()