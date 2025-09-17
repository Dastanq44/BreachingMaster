@echo off
:: Запуск Python скрипта от имени администратора
powershell -Command "Start-Process python 'my_script.py' -Verb runAs"
