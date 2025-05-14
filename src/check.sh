#!/bin/bash

echo "Vérification du port 5000..."
echo

sudo lsof -i :5000 || echo "Le port 5000 n'est plus utilisé."

echo
echo "📦 Liste des processus Flask actifs (si existants) :"
ps aux | grep '[p]ython.*flask\|run.py'
