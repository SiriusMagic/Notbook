"""Lista de referencia: palabras silabeadas a mano según las reglas de la RAE.

Sirve de «respuesta correcta» independiente para comparar segmentadores. Se
evitan palabras con prefijo dudoso (sub-, des-, in-) y el grupo «tl», que
admiten dos silabeos válidos.
"""

REFERENCIA: dict[str, str] = {
    # sílaba directa y grupos inseparables
    "casa": "ca-sa", "mesa": "me-sa", "palabra": "pa-la-bra", "otro": "o-tro", "libro": "li-bro",
    "hablar": "ha-blar", "flor": "flor", "fruta": "fru-ta", "grande": "gran-de", "cristal": "cris-tal",
    "problema": "pro-ble-ma", "tecla": "te-cla", "madre": "ma-dre", "sobre": "so-bre", "inglés": "in-glés",
    "playa": "pla-ya", "abrazo": "a-bra-zo", "aplauso": "a-plau-so", "ofrecer": "o-fre-cer", "agradable": "a-gra-da-ble",
    # dígrafos y letras especiales
    "perro": "pe-rro", "carro": "ca-rro", "calle": "ca-lle", "pollo": "po-llo", "chocolate": "cho-co-la-te",
    "noche": "no-che", "muchacho": "mu-cha-cho", "niño": "ni-ño", "mañana": "ma-ña-na", "querer": "que-rer",
    "guerra": "gue-rra", "guitarra": "gui-ta-rra", "pingüino": "pin-güi-no", "vergüenza": "ver-güen-za",
    "examen": "e-xa-men", "taxi": "ta-xi", "yema": "ye-ma", "rey": "rey", "hoy": "hoy", "muy": "muy", "leyes": "le-yes",
    "ahora": "a-ho-ra", "prohibir": "pro-hi-bir", "búho": "bú-ho", "zanahoria": "za-na-ho-ria",
    # dos y más consonantes entre vocales
    "carta": "car-ta", "campo": "cam-po", "mundo": "mun-do", "perla": "per-la", "alto": "al-to", "puerta": "puer-ta",
    "instante": "ins-tan-te", "constante": "cons-tan-te", "transporte": "trans-por-te", "construir": "cons-truir",
    "abstracto": "abs-trac-to", "obstruir": "obs-truir", "empresa": "em-pre-sa", "hombre": "hom-bre",
    "nombre": "nom-bre", "entrada": "en-tra-da", "siempre": "siem-pre", "compra": "com-pra", "excelente": "ex-ce-len-te",
    "extraño": "ex-tra-ño", "inspector": "ins-pec-tor", "monstruo": "mons-truo", "pregunta": "pre-gun-ta",
    "producto": "pro-duc-to", "objeto": "ob-je-to", "acción": "ac-ción", "digno": "dig-no", "técnico": "téc-ni-co",
    # diptongos y triptongos
    "ciudad": "ciu-dad", "cuidado": "cui-da-do", "bueno": "bue-no", "tierra": "tie-rra", "aire": "ai-re",
    "causa": "cau-sa", "peine": "pei-ne", "deuda": "deu-da", "oigo": "oi-go", "agua": "a-gua", "nuevo": "nue-vo",
    "canción": "can-ción", "estudiar": "es-tu-diar", "Uruguay": "U-ru-guay", "buey": "buey", "averiguáis": "a-ve-ri-guáis",
    "limpiéis": "lim-piéis", "cuota": "cuo-ta", "viaje": "via-je", "precio": "pre-cio",
    # hiatos
    "poeta": "po-e-ta", "leer": "le-er", "caos": "ca-os", "teatro": "te-a-tro", "museo": "mu-se-o",
    "país": "pa-ís", "río": "rí-o", "día": "dí-a", "maíz": "ma-íz", "baúl": "ba-úl", "frío": "frí-o",
    "aéreo": "a-é-re-o", "océano": "o-cé-a-no", "creer": "cre-er", "zoológico": "zo-o-ló-gi-co",
    "reír": "re-ír", "actúa": "ac-tú-a", "envío": "en-ví-o", "sonreír": "son-re-ír", "aldea": "al-de-a",
    # palabras de reseñas
    "recomiendo": "re-co-mien-do", "llegó": "lle-gó", "rápido": "rá-pi-do", "calidad": "ca-li-dad",
    "funciona": "fun-cio-na", "perfecto": "per-fec-to", "encantó": "en-can-tó", "pésimo": "pé-si-mo",
    "devolución": "de-vo-lu-ción", "tamaño": "ta-ma-ño", "batería": "ba-te-rí-a", "pantalla": "pan-ta-lla",
    "estrellas": "es-tre-llas", "material": "ma-te-rial", "precioso": "pre-cio-so",
}
