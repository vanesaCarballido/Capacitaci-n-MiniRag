# Vanesa Carballido : capacitación RAG

_Proyecto:_ 

 Es un mini RAG que recibe una frase de un usuario (ingresada en el momento) y devuelve dos citas relacionada incluyendo los autores.

_________________________________

## 💚  ¿Qué hace mi proyecto de RAG? 

✨ Toma una frase ingresada por el usuario, la transforma en un vector numérico (embedding) para que una inteligencia artificial (en este caso Gemini) pueda entender su significado

✨ Luego busca en una base de datos de frases filosóficas (en este caso ChromaDB) aquellas que sean más parecidas en su sentido

✨ Finalmente devuelve las 2 frases más similares, incluyendo el nombre del filósofo que las dijo.
_______________

## 💜 Ejecución por consola:
- Clonar repositorio

- Instalar requeriments, escribir en consola: "pip install -r requirements.txt"

- Correr el main: "py main.py"
_______________

## 🌸 Pasos luego de ejecutar el programa main.py: 

- Al terminar de ejecutar todos los pasos se tendría que ejecutar correctamente y mostrar esto:

<img width="600" height="131" alt="Captura de pantalla 2025-07-26 162256" src="https://github.com/user-attachments/assets/6b603bbc-2227-49fa-95e4-55f20aad362f" />

- Luego se ingresa la frase, dando como resultado:


<img width="400" height="200" alt="Captura de pantalla 2025-07-26 162812" src="https://github.com/user-attachments/assets/620119d1-2bae-4b4f-a110-6065feba29d5" />

