# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:23:04 2023

@author: lcenci1
"""

# Importando as bibliotecas necessárias
import requests
from bs4 import BeautifulSoup

# Fazendo uma requisição HTTP para o site desejado
url = "https://pt.wikipedia.org/wiki/Monge_Jo%C3%A3o_Maria"
response = requests.get(url)

# Criando um objeto BeautifulSoup para fazer o parsing do HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Encontrando os parágrafos desejados pelo nome da classe
paragrafos = soup.find_all('p')

# Iterando sobre os parágrafos e imprimindo o conteúdo de cada um
for p in paragrafos:
    print(p.text)
