#include <iostream>

#include <QApplication>
#include <QMainWindow>
#include "rl/MappedEnvironment.h"

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);
    QMainWindow w;
    w.show();
    return a.exec();
}