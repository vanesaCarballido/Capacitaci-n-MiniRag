
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

#Configuración de Gemini con mi key:
geminiApiKey = "AIzaSyBYWF_lqUEDttKs7WkpXftpAt0vOeZwNq8"
genai.configure(api_key=geminiApiKey) #siempre api_key, eso no se cambia, el nombre de la variable de nuestra key si

#Configuración de ChromaDB:
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("RagDeFilosofia")

#Cargar modelo de embeddings para convertir frases en vectores:
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#Frases de los filósofos
frases = [
    #Aristóteles:
    {"frase": "El alma nunca piensa sin una imagen.", "autor": "Aristóteles"},
    {"frase": "La amistad es un alma que habita en dos cuerpos; un corazón que habita en dos almas.", "autor": "Aristóteles"},
    {"frase": "Somos lo que hacemos repetidamente. La excelencia, entonces, no es un acto, sino un hábito.", "autor": "Aristóteles"},
    {"frase": "El hombre es por naturaleza un animal político.", "autor": "Aristóteles"},
    {"frase": "La inteligencia consiste no sólo en el conocimiento, sino también en la destreza de aplicar los conocimientos en la práctica.", "autor": "Aristóteles"},
    {"frase": "La finalidad del arte es dar cuerpo a la esencia secreta de las cosas, no copiar su apariencia.", "autor": "Aristóteles"},
    {"frase": "La educación es el mejor provisionamiento para la vejez.", "autor": "Aristóteles"},
    {"frase": "La esperanza es el sueño del hombre despierto.", "autor": "Aristóteles"},
    {"frase": "Conocerse a sí mismo es el principio de toda sabiduría.", "autor": "Aristóteles"},
    {"frase": "El sabio no dice todo lo que piensa, pero siempre piensa todo lo que dice.", "autor": "Aristóteles"},

    #Nietzsche:
    {"frase": "Dios ha muerto. Y nosotros lo hemos matado.", "autor": "Nietzsche"},
    {"frase": "Quien tiene un porqué para vivir puede soportar casi cualquier cómo.", "autor": "Nietzsche"},
    {"frase": "Lo que no me mata, me hace más fuerte.", "autor": "Nietzsche"},
    {"frase": "No hay hechos, solo interpretaciones.", "autor": "Nietzsche"},
    {"frase": "La esperanza es el peor de los males, pues prolonga el tormento del hombre.", "autor": "Nietzsche"},
    {"frase": "El individuo ha luchado siempre para no ser absorbido por la tribu.", "autor": "Nietzsche"},
    {"frase": "El destino de los hombres está hecho de momentos felices, toda la vida los tiene, pero no de épocas felices.", "autor": "Nietzsche"},
    {"frase": "No puedo creer en un Dios que quiera ser alabado todo el tiempo.", "autor": "Nietzsche"},
    {"frase": "La moral es la mejor de todas las mentiras.", "autor": "Nietzsche"},
    {"frase": "La serpiente que no puede mudar su piel, perece. Lo mismo ocurre con los espíritus que se les impide cambiar de opinión.", "autor": "Nietzsche"},
    {"frase": "El hombre, en su orgullo, creó a Dios a su imagen y semejanza.", "autor": "Nietzsche"},
    {"frase": "La verdad es fea: poseemos el arte para no morir por la verdad.", "autor": "Nietzsche"},
    {"frase": "Es necesario llevar en uno mismo un caos para poder dar a luz una estrella danzante.", "autor": "Nietzsche"},
    {"frase": "Quien combate con monstruos debe tener cuidado de no convertirse en uno de ellos.", "autor": "Nietzsche"},
    {"frase": "El mayor enemigo de la verdad no es la mentira, sino la convicción.", "autor": "Nietzsche"},

    #Sócrates
    {"frase": "Solo sé que no sé nada.", "autor": "Socrates"},
    {"frase": "Una vida sin examen no merece ser vivida.", "autor": "Socrates"},
    {"frase": "Conócete a ti mismo.", "autor": "Socrates"},
    {"frase": "El comienzo de la sabiduría es la definición de los términos.", "autor": "Socrates"},
    {"frase": "Prefiero el conocimiento a la riqueza, porque el primero es eterno, mientras que la riqueza puede perderse.", "autor": "Socrates"},
    {"frase": "El secreto de la felicidad no se encuentra en buscar más, sino en desarrollar la capacidad de disfrutar con menos.", "autor": "Socrates"},
    {"frase": "El mayor de todos los misterios es el hombre.", "autor": "Socrates"},
    {"frase": "No hagas a otros lo que te enfadaría si te lo hicieran a ti.", "autor": "Socrates"},
    {"frase": "La educación es el encendido de una llama, no el llenado de un recipiente.", "autor": "Socrates"},
    {"frase": "Habla para que yo te conozca.", "autor": "Socrates"},
    {"frase": "La verdadera sabiduría está en reconocer la propia ignorancia.", "autor": "Socrates"},
    {"frase": "Es peor cometer una injusticia que sufrirla.", "autor": "Socrates"},
    {"frase": "El alma se cura con ciertos encantamientos, y esos encantamientos son las palabras.", "autor": "Socrates"},
    {"frase": "Cuando el debate se pierde, la calumnia es la herramienta del perdedor.", "autor": "Socrates"},
    {"frase": "La muerte podría ser la mayor de las bendiciones humanas.", "autor": "Socrates"},

    #Kant:
    {"frase": "Obra sólo según aquella máxima por la cual puedas querer que al mismo tiempo se convierta en ley universal.", "autor": "Kant"},
    {"frase": "La libertad es aquella facultad que aumenta la utilidad de todas las demás facultades.", "autor": "Kant"},
    {"frase": "La moral no es la doctrina de cómo hacernos felices, sino de cómo debemos llegar a ser dignos de la felicidad.", "autor": "Kant"},
    {"frase": "El sabio puede cambiar de opinión. El necio, nunca.", "autor": "Kant"},
    {"frase": "La ilustración es la salida del hombre de su autoculpable minoría de edad.", "autor": "Kant"},

    #Platón:
    {"frase": "El alma del hombre es inmortal e imperecedera.", "autor": "Platón"},
    {"frase": "La música es para el alma lo que la gimnasia para el cuerpo.", "autor": "Platón"},
    {"frase": "El conocimiento verdadero viene del interior.", "autor": "Platón"},
    {"frase": "La ignorancia es la semilla del mal.", "autor": "Platón"},
    {"frase": "No hay hombre tan cobarde a quien el amor no haga valiente.", "autor": "Platón"},

    #Descartes:
    {"frase": "Pienso, luego existo.", "autor": "Descartes"},
    {"frase": "Para investigar la verdad es preciso dudar, en cuanto sea posible, de todas las cosas.", "autor": "Descartes"},
    {"frase": "No hay nada repartido de modo más equitativo que la razón: todo el mundo está convencido de tener suficiente.", "autor": "Descartes"},
    {"frase": "Conquistarme a mí mismo, más que al mundo.", "autor": "Descartes"},
    {"frase": "Es prudente no fiarse por entero de quienes nos han engañado una vez.", "autor": "Descartes"}

]
    

#Insertar frases en ChromaDB, se ingresa los vectores y ChromaDB compara con las de "frases":
for i, frase in enumerate(frases):
    embedding = embedder.encode(frase["frase"]).tolist()
    collection.add(
        ids=[f"frase_{i}"],
        documents=[frase["frase"]],
        metadatas=[{"autor": frase["autor"]}],
        embeddings=[embedding]
    )

#Ingreso de la frase manual del usuario:
fraseIngresada = input("Ingrese su frase: ")
if fraseIngresada==" ":
    print("Tenes que ingresar por lo menos una palabra.")
    exit()
    
arreglarFrase = embedder.encode(fraseIngresada).tolist()
resultados = collection.query(query_embeddings=[arreglarFrase], n_results=3)

#Mostrar los resultados obtenidos:
print("\nLa frase que ingresó el usuario es:", fraseIngresada)
print('\t')
print("\nLas frases más similares en la lista de frases de filósofos son:")
print('\t')
for frase, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
    print("........................")
    print(f'Frase: {frase}')
    print(f'\nFilosofo: {meta["autor"]}')
   
#Frase del sabio:
contexto = "\n".join([
    f"- {frase} ({meta['autor']})"
    for frase, meta in zip(resultados["documents"][0], resultados["metadatas"][0])
])

#Mensaje para gemini
prompt = f"""Eres un sabio antiguo y una persona te dice: "{fraseIngresada}". 
Contéstale usando las frases citadas en: {contexto}, mencionando el autor que corresponde a cada frase utilizada.
Que el mensaje sea reflexivo, claro, profundo y breve."""

#Enviar el prompt a gemini
ejemplo = genai.GenerativeModel("gemini-1.5-flash")
respuestaDeGemini = ejemplo.generate_content(prompt)

# Mostrar la respuesta del sabio
print("______________________________________________________________________")
print("\nLa respuesta del sabio:\n")
print(respuestaDeGemini.text)
