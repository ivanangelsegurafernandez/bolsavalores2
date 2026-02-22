import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pygame
import subprocess
import os
from datetime import datetime
import json
import asyncio
import websockets
import platform
import threading
import time
import sys
import psutil  # Para lock file PID checking

# Estilos constantes - Mejorados para belleza visual
COLOR_FONDO_TEXTO = "#000000"  # Fondo negro más oscuro
COLOR_TEXTO_INTRO = "#FFFFFF"  # Blanco puro para máxima legibilidad
COLOR_TITULO = "#00FFAA"  # Verde más vibrante
COLOR_SUBTITULO = "#4DD0E1"  # Azul más suave
COLOR_ETIQUETA = "#1DE9B6"  # Verde cyan para etiquetas
COLOR_FONDO_ETIQUETA = "#263238"  # Fondo gris oscuro para contraste
COLOR_FIRMA = "#FFEB3B"  # Amarillo oro brillante
COLOR_BOTON = "#2E7D32"  # Verde bosque para botón, más profesional
COLOR_SOMBRA = "#000000"  # Negro para sombras
COLOR_FONDO_ENTRADA = "#FAFAFA"  # Blanco suave
COLOR_TEXTO_ENTRADA = "#212121"  # Gris oscuro para texto entrada
COLOR_BORDE_ENTRADA = "#4FC3F7"  # Azul claro para bordes
COLOR_FOOTER = "#1A1A1A80"  # Semi-transparente para barra inferior
FUENTE_TITULO = ("Impact", 72, "bold")  # Tamaño duplicado
FUENTE_SUBTITULO = ("Arial", 18, "italic")  # Comprimido para más espacio
FUENTE_ETIQUETA = ("Arial", 18, "bold")
FUENTE_ENTRADA = ("Arial", 16)
FUENTE_FIRMA = ("Georgia", 16, "italic")  # Mayor tamaño e itálica
FUENTE_HORA = ("Consolas", 14)

# Estado global
processes = []
is_starting = False
is_running = False
last_enter_ms = 0
ENTER_DEBOUNCE_MS = 800
LOCK_FILE = "evabot.lock"
SCRIPTS_TO_LAUNCH = [
    "5R6M-1-2-4-8-16.py",
    "botttt45-1-2-4-8-16-32.py",
    "botttt46-1-2-4-8-16-32.py",
    "botttt47-1-2-4-8-16-32.py",
    "botttt48-1-2-4-8-16-32.py",
    "botttt49-1-2-4-8-16-32.py",
    "botttt50-1-2-4-8-16-32.py",
]


def get_scaled_fonts(screen_w, screen_h):
    """Calcula fuentes y tamaños adaptativos para distintas pantallas."""
    scale = min(screen_w / 1920, screen_h / 1080)
    scale = max(0.65, min(scale, 1.15))

    return {
        "titulo": ("Impact", max(36, int(72 * scale)), "bold"),
        "etiqueta": ("Arial", max(11, int(18 * scale)), "bold"),
        "entrada": ("Arial", max(10, int(16 * scale))),
        "firma": ("Georgia", max(9, int(16 * scale)), "italic"),
        "hora": ("Consolas", max(9, int(14 * scale))),
        "boton": ("Arial", max(12, int(20 * scale)), "bold"),
        "intro": ("Arial", max(10, int(18 * scale))),
    }


def get_python_command():
    """Obtiene el intérprete actual para evitar fallos entre laptops."""
    return sys.executable or "python"

def aplicar_estilo_label(texto, fuente, fg, bg=None):
    """Aplica estilos consistentes a un Label."""
    kwargs = {"text": texto, "font": fuente, "fg": fg, "padx": 10, "pady": 5}
    if bg:
        kwargs["bg"] = bg
    return tk.Label(root, **kwargs)

async def validar_token_ws(token):
    """Valida un token con la API de Deriv y devuelve detalles de la cuenta o None si es inválido."""
    try:
        async with websockets.connect("wss://ws.binaryws.com/websockets/v3?app_id=1089") as ws:
            await ws.send(json.dumps({"authorize": token}))
            response = json.loads(await ws.recv())
            if "error" in response:
                raise Exception(response["error"]["message"])
            auth = response.get("authorize", {})
            return {
                "loginid": auth.get("loginid"),
                "fullname": auth.get("fullname"),
                "balance": auth.get("balance"),
                "is_virtual": auth.get("is_virtual") == 1
            }
    except Exception as e:
        messagebox.showerror("Error de Validación", f"Token inválido: {str(e)}")
        return None

def load_previous_tokens():
    """Carga tokens previos desde tokens_ingresados.json si existe."""
    try:
        with open("tokens_ingresados.json", "r", encoding="utf-8") as f:
            tokens = json.load(f)
            entry_token_demo.delete(0, tk.END)
            entry_token_demo.insert(0, tokens.get("demo", ""))
            entry_token_real.delete(0, tk.END)
            entry_token_real.insert(0, tokens.get("real", ""))
    except FileNotFoundError:
        pass

def crear_lock():
    """Crea un archivo de lock con el PID para evitar múltiples instancias."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                pid = int(f.read().strip())
                if pid and pid != os.getpid() and psutil.pid_exists(pid):
                    messagebox.showwarning("EVA BOT", "Ya hay una instancia ejecutándose.")
                    return False
        except:
            pass
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True

def liberar_lock():
    """Elimina el archivo de lock al cerrar."""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except:
        pass

async def start_eva_bot():
    """Valida, guarda los tokens y lanza los bots, mostrando resultados en la GUI y bloqueando la interfaz."""
    global processes, is_starting, is_running
    if is_starting or is_running:
        return
    is_starting = True
    start_button.config(text="Iniciando…", state="disabled")
    
    # Crear lock
    if not crear_lock():
        is_starting = False
        start_button.config(text="✅ Iniciar sistema", state="normal")
        return
    
    token1 = entry_token_demo.get().strip()
    token2 = entry_token_real.get().strip()
    
    # Validación básica
    if not token1 or not token2:
        validation_label.config(text="⚠️ Error: campo vacío", fg="red")
        is_starting = False
        start_button.config(text="✅ Iniciar sistema", state="normal")
        liberar_lock()
        return
    if len(token1) < 5 or len(token2) < 5:
        validation_label.config(text="⚠️ Error: Los tokens deben tener al menos 5 caracteres.", fg="red")
        is_starting = False
        start_button.config(text="✅ Iniciar sistema", state="normal")
        liberar_lock()
        return
    
    # Validar tokens con la API de Deriv
    result1 = await validar_token_ws(token1)
    result2 = await validar_token_ws(token2)
    
    # Mostrar resultados de validación
    validation_text = ""
    if result1:
        validation_text += f"🟢 Token demo válido | Cuenta: {result1['loginid']} | Saldo: {result1['balance']}\n"
    else:
        validation_text += "🔴 Token demo inválido\n"
    if result2:
        validation_text += f"🟢 Token real válido | Cuenta: {result2['loginid']} | Saldo: {result2['balance']}"
    else:
        validation_text += "🔴 Token real inválido"
    
    validation_label.config(text=validation_text, fg="white" if result1 and result2 else "red")
    
    # Guardar tokens si ambos son válidos y tienen el tipo correcto
    if result1 and result2:
        if result1["is_virtual"] and not result2["is_virtual"]:
            token_demo, token_real = token1, token2
        elif not result1["is_virtual"] and result2["is_virtual"]:
            token_demo, token_real = token2, token1
        else:
            validation_label.config(text="⚠️ Error: Los tokens deben ser uno DEMO y uno REAL.", fg="red")
            is_starting = False
            start_button.config(text="✅ Iniciar sistema", state="normal")
            liberar_lock()
            return
        
        # Guardar en tokens_ingresados.json
        try:
            with open("tokens_ingresados.json", "w", encoding="utf-8") as f:
                json.dump({"demo": token_demo, "real": token_real}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            validation_label.config(text=f"⚠️ Error: no se pudo guardar tokens_ingresados.json: {str(e)}", fg="red")
            is_starting = False
            start_button.config(text="✅ Iniciar sistema", state="normal")
            liberar_lock()
            return
        
        # Guardar en tokens_usuario.txt
        try:
            with open("tokens_usuario.txt", "w", encoding="utf-8") as f:
                f.write(f"{token_demo}\n{token_real}\n")
        except Exception as e:
            validation_label.config(text=f"⚠️ Error: no se pudo guardar tokens_usuario.txt: {str(e)}", fg="red")
            is_starting = False
            start_button.config(text="✅ Iniciar sistema", state="normal")
            liberar_lock()
            return
        
        # Inicializar token_actual.txt con REAL:none
        try:
            with open("token_actual.txt", "w", encoding="utf-8") as f:
                f.write("REAL:none")
            validation_label.config(text="✅ Tokens guardados", fg="white")
        except Exception as e:
            validation_label.config(text=f"⚠️ Error: no se pudo guardar token_actual.txt: {str(e)}", fg="red")
            is_starting = False
            start_button.config(text="✅ Iniciar sistema", state="normal")
            liberar_lock()
            return
        
        # Detener música
        pygame.mixer.music.stop()
        
        # Lanzar los scripts
        processes = []
        scripts = SCRIPTS_TO_LAUNCH
        validation_text = "🚀 Lanzando bots...\n"
        python_cmd = get_python_command()
        for script in scripts:
            script_path = os.path.join(os.path.dirname(__file__), script)
            if not os.path.exists(script_path):
                validation_label.config(text=f"⚠️ Error: El archivo {script} no se encuentra.", fg="red")
                is_starting = False
                start_button.config(text="✅ Iniciar sistema", state="normal")
                liberar_lock()
                return
            try:
                kwargs = {"cwd": os.path.dirname(__file__), "close_fds": True}
                if platform.system() == "Windows" and os.getenv("EVA_OPEN_CONSOLES", "0") == "1":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

                p = subprocess.Popen([python_cmd, script_path], **kwargs)
                processes.append(p)
                if script == "5R6M-1-2-4-8-16.py":
                    validation_text += "🚀 Programa maestro iniciado\n"
                else:
                    bot_num = script.split('botttt')[1].split('-')[0]
                    validation_text += f"⚙️ Bot {bot_num} activo\n"
            except Exception as e:
                validation_label.config(text=f"⚠️ Error: No se pudo lanzar {script}: {str(e)}", fg="red")
                is_starting = False
                start_button.config(text="✅ Iniciar sistema", state="normal")
                liberar_lock()
                return
        
        validation_label.config(text=validation_text + "🟢 Todos los bots ejecutándose…", fg="white")
        is_running = True
        is_starting = False
        start_button.config(text="En ejecución...", state="disabled")
        stop_button.config(text="🛑 SALIR", font=("Arial", 28, "bold"), bg="#D32F2F", state="normal")
        entry_token_demo.config(state="disabled")
        entry_token_real.config(state="disabled")

def stop_and_close():
    """Termina los procesos del bot y cierra la ventana."""
    global processes, is_running, is_starting
    for p in processes:
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    is_running = False
    is_starting = False
    liberar_lock()
    root.destroy()

def actualizar_hora(canvas, hora_label, ancho, alto):
    """Actualiza la hora y fecha en la esquina inferior derecha cada segundo con sombra."""
    ahora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    hora_label.config(text=ahora)
    canvas.itemconfig("hora_sombra", text=ahora)
    root.after(1000, lambda: actualizar_hora(canvas, hora_label, ancho, alto))

def on_enter(event=None):
    """Maneja la tecla Enter con debounce según el estado."""
    global last_enter_ms, is_starting, is_running
    now = int(time.monotonic() * 1000)
    if (now - last_enter_ms) < ENTER_DEBOUNCE_MS:
        return
    last_enter_ms = now
    if is_running:
        stop_and_close()
    elif not is_starting:
        threading.Thread(target=lambda: asyncio.run(start_eva_bot()), daemon=True).start()

def mostrar_presentacion():
    """Configura la interfaz gráfica de bienvenida."""
    global entry_token_demo, entry_token_real, bg_img, logo_img, validation_label, start_button, stop_button

    # Configurar ventana con tamaño adaptable
    ancho = root.winfo_screenwidth()
    alto = root.winfo_screenheight()
    fonts = get_scaled_fonts(ancho, alto)
    root.minsize(800, 600)
    root.maxsize(ancho, alto)
    root.geometry(f"{ancho}x{alto}")
    root.resizable(True, True)

    # Cargar imagen de fondo con PIL
    try:
        img = Image.open(os.path.join(os.path.dirname(__file__), "evabot.png"))
        ratio_img = img.width / img.height
        ratio_win = ancho / alto
        if ratio_img > ratio_win:
            new_width = ancho
            new_height = int(ancho / ratio_img)
        else:
            new_height = alto
            new_width = int(alto * ratio_img)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        bg_img = ImageTk.PhotoImage(img)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar la imagen de fondo evabot.png: {str(e)}")
        bg_img = None

    # Cargar logo
    logo_img = None
    try:
        logo = Image.open(os.path.join(os.path.dirname(__file__), "eva_logo.png"))
        logo = logo.resize((120, 120), Image.LANCZOS)
        logo_img = ImageTk.PhotoImage(logo)
    except:
        pass

    # Usar Canvas para fondo y widgets
    canvas = tk.Canvas(root, width=ancho, height=alto)
    canvas.pack(fill="both", expand=True)
    if bg_img:
        x_offset = (ancho - new_width) // 2
        y_offset = (alto - new_height) // 2
        canvas.create_image(x_offset, y_offset, image=bg_img, anchor="nw")
    if logo_img:
        canvas.create_image(ancho - 130, 10, image=logo_img, anchor="ne")

    # Texto de presentación
    margen_x = ancho * 0.1
    margen_y = 50
    intro_text = (
        "📘 BIENVENIDO A EVA BOT – ESTRATEGIA MARTINGALA INTELIGENTE\n\n"
        "Este programa ha sido diseñado con un solo objetivo:\n"
        "💡 convertir pequeñas inversiones en ingresos constantes, de forma automática,\n"
        "simple y al alcance de cualquier persona.\n\n"
        "EVA BOT aplica una lógica clara, probada y funcional que ha sido estudiada\n"
        "a lo largo de cientos (incluso miles) de operaciones reales.\n\n"
        "⚙️ EVA BOT no requiere que seas un experto en trading ni en finanzas.\n"
        "Está pensado para personas que buscan automatizar sus ingresos con una\n"
        "herramienta confiable, intuitiva y lista para trabajar por ti."
    )
    font_size = fonts["intro"][1]
    ancho_texto = min(int(ancho * 0.78), 1200)
    text_id = canvas.create_text(ancho // 2, margen_y + 30, text=intro_text, font=fonts["intro"], fill="#FFFFFF", justify="center", width=ancho_texto, anchor="n")

    # Medir bbox del texto y dibujar rectángulo detrás con padding
    padding_x = 20
    padding_y = 10
    bbox = canvas.bbox(text_id)
    rect_id = canvas.create_rectangle(bbox[0] - padding_x, bbox[1] - padding_y, bbox[2] + padding_x, bbox[3] + padding_y, fill="#000000", stipple="gray50", outline="")
    canvas.tag_lower(rect_id)

    # Título
    title_offset = 217
    if alto > 800:
        title_offset += int(alto * 0.01)
    canvas.create_text(ancho // 2, bbox[3] + title_offset, text="EVA BOT", font=fonts["titulo"], fill=COLOR_TITULO)

    # Banda semitransparente detrás del bloque de tokens
    desplazamiento_px = int(root.winfo_fpixels('0.5c'))  # ~0.5 cm ≈ 40 px en 96 DPI
    y_base = bbox[3] + 100 - desplazamiento_px  # Subir bloque de tokens
    x_centro = ancho // 2
    token_block_height = 260  # Aproximado para cubrir labels, entradas y validación
    canvas.create_rectangle(x_centro - 400, y_base - 20, x_centro + 400, y_base + token_block_height, fill="#000000", stipple="gray50", outline="")
    
    # Etiquetas y entradas para tokens
    label_demo = aplicar_estilo_label("Token DEMO:", fonts["etiqueta"], COLOR_ETIQUETA, COLOR_FONDO_ETIQUETA)
    canvas.create_window(x_centro, y_base, window=label_demo)

    entry_width = max(34, min(70, int(ancho / 24)))
    entry_token_demo = tk.Entry(root, width=entry_width, font=fonts["entrada"], bg=COLOR_FONDO_ENTRADA, fg=COLOR_TEXTO_ENTRADA,
                                highlightbackground=COLOR_BORDE_ENTRADA, highlightthickness=3, bd=1, relief="flat")
    canvas.create_window(x_centro, y_base + 50, window=entry_token_demo)

    label_real = aplicar_estilo_label("Token REAL:", fonts["etiqueta"], COLOR_ETIQUETA, COLOR_FONDO_ETIQUETA)
    canvas.create_window(x_centro, y_base + 100, window=label_real)

    entry_token_real = tk.Entry(root, width=entry_width, font=fonts["entrada"], bg=COLOR_FONDO_ENTRADA, fg=COLOR_TEXTO_ENTRADA,
                                highlightbackground=COLOR_BORDE_ENTRADA, highlightthickness=3, bd=1, relief="flat", show="*")
    canvas.create_window(x_centro, y_base + 150, window=entry_token_real)

    # Área para resultados de validación
    validation_label = aplicar_estilo_label("", fonts["entrada"], "white", COLOR_FONDO_ETIQUETA)
    canvas.create_window(x_centro, y_base + 200, window=validation_label)

    # Botón de inicio
    start_button = tk.Button(root, text="✅ Iniciar sistema", font=fonts["boton"],
                             bg=COLOR_BOTON, fg="white", relief="flat", bd=0, padx=20, pady=10,
                             command=lambda: threading.Thread(target=lambda: asyncio.run(start_eva_bot()), daemon=True).start())
    canvas.create_window(x_centro, y_base + 150, window=start_button)  # Reducir espacio vertical

    # Botón para salir y cerrar bots
    stop_button = tk.Button(root, text="Salir y cerrar bots", font=fonts["boton"],
                            bg="#D32F2F", fg="white", relief="flat", bd=0, padx=20, pady=10,
                            command=stop_and_close, state="disabled")
    canvas.create_window(x_centro, y_base + 210, window=stop_button)  # Reducir espacio vertical

    # Vincular teclas
    root.bind("<Return>", on_enter)
    entry_token_real.bind("<Return>", on_enter)
    root.bind("<Escape>", lambda e: messagebox.askyesno("EVA BOT", "Confirmar salida?") and stop_and_close())

    # Cargar tokens previos
    load_previous_tokens()

    # Barra inferior
    footer_y = alto - 160
    canvas.create_rectangle(0, footer_y, ancho, alto, fill=COLOR_FONDO_ETIQUETA, stipple="gray50")
    canvas.create_text(12, footer_y + 67, text="by Ing Biomedico Ivan Angel Segura Fernandez", font=fonts["firma"], fill=COLOR_SOMBRA, anchor="sw")
    firma_label = aplicar_estilo_label("by Ing Biomedico Ivan Angel Segura Fernandez", fonts["firma"], COLOR_FIRMA, COLOR_FONDO_ETIQUETA)
    canvas.create_window(10, footer_y + 67, window=firma_label, anchor="sw")

    # Hora y fecha
    hora_label = tk.Label(root, font=fonts["hora"], bg=COLOR_FONDO_ETIQUETA, fg="white", padx=10, pady=5)
    canvas.create_window(ancho - 10, footer_y + 30, window=hora_label, anchor="se")
    canvas.create_text(ancho - 12, footer_y + 32, text="", font=fonts["hora"], fill=COLOR_SOMBRA, anchor="se", tag="hora_sombra")
    actualizar_hora(canvas, hora_label, ancho, alto)

    # Manejar cierre de ventana
    root.protocol("WM_DELETE_WINDOW", stop_and_close)

    return canvas, bg_img, logo_img

if __name__ == "__main__":
    # Inicializar pygame para música
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), "EVABOT.mp3"))
        pygame.mixer.music.set_volume(0.2)
        pygame.mixer.music.play(-1)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo de audio EVABOT.mp3: {str(e)}")

    # Crear ventana principal
    root = tk.Tk()
    root.title("EVA BOT")
    canvas, bg_img, logo_img = mostrar_presentacion()
    root.mainloop()
