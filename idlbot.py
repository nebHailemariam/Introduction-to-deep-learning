import discord
from discord.ext import commands, tasks
import datetime
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import logging
import sys
import asyncio  # Ensure you import asyncio

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize bot intents and bot object
intents = discord.Intents.default()
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

user_responsibilities = {
    "kshapovalenko": [1134188563236409404, ["Monday", "Tuesday"]],
    # "Miya Sylvester": [100000000, []],
    "yosh3441": [225488327036370944, ["Saturday", "Sunday"]],
    "puruboii": [582931106605563904, ["Saturday", "Sunday"]],
    "shravz.": [847319475040157768, ["Thursday", "Friday"]],
    "hihicrab": [747430516810317864, ["Friday", "Saturday"]],
    "massa.baali": [1191394865750216807, ["Wednesday", "Friday"]],
    "vedantsingh_56957": [1181754582414675989, ["Tuesday", "Thursday"]],
    "sadrishya": [617383770500235276, ["Friday", "Sunday"]],
    "dornub": [1056406329092218890, ["Wednesday", "Sunday"]],
    "vvoforreal": [756414206349410314, ["Wednesday", "Saturday"]],
    "_ishita_gupta_": [768524190097342466, ["Monday", "Tuesday"]],
    "skachroo": [467046263729160202, ["Tuesday", "Wednesday"]],
    "shreyjain711": [1280280367940632667, ["Wednesday", "Saturday"]],
    "fnzabaki": [1308581724107571351, ["Tuesday", "Thursday"]],
    "c_muthee_31148": [1257238593856868362, ["Tuesday", "Sunday"]],
    "ahmedissahtahiru": [1206316006041976893, ["Saturday", "Sunday"]],
    "shubhamsingh_15534": [1308615486543499315, ["Monday", "Wednesday"]],
    "tanettech": [1311026963040960563, ["Thursday", "Friday"]],
    "johnliu090203": [1311369823468388363, ["Thursday", "Friday"]],
    "ema1545": [744141074779471919, ["Monday", "Wednesday"]],
    # "Harshith Arun Kumar": [100000000, []],
    "olatunji_damilare": [1259328537266094162, ["Tuesday", "Wednesday"]],
    "ebyau": [947067276732014613, ["Monday", "Thursday"]],
    "peterwauyo_85654": [1247696351425466498, ["Tuesday", "Thursday"]],
}


@tasks.loop(hours=24)
async def send_reminder():
    current_day = datetime.datetime.now().strftime("%A")
    channel = bot.get_channel(1306667280209281093)

    if channel is None:
        logger.error("Channel not found!")
        return  # Exit the function if channel is None

    reminder_message = "Reminder: "

    responsible_users = []
    for user, info in user_responsibilities.items():
        user_id, days = info
        if current_day in days:
            responsible_users.append(f"<@{user_id}>")

    if responsible_users:
        reminder_message += (
            " ".join(responsible_users) + ", it's your Piazza responsibility day today!"
        )
        await channel.send(reminder_message)
    else:
        await channel.send("No one has a responsibility today.")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"message": "Bot is running!"}')
        except ConnectionResetError:
            logger.warning("Connection reset by peer, ignoring.")
        except Exception as e:
            logger.error(f"Unexpected error in HTTP server: {e}")


def run_http_server():
    try:
        server = ThreadingHTTPServer(("0.0.0.0", 9000), SimpleHTTPRequestHandler)
        logger.info("Starting HTTP server on port 9000...")
        server.serve_forever()
    except Exception as e:
        logger.error(f"HTTP server failed: {e}")
        sys.exit(1)  # Exit to restart


@bot.event
async def on_ready():
    try:
        logger.info(f"Logged in as {bot.user}.")
        send_reminder.start()

        # Start HTTP server in a separate thread
        http_server_thread = threading.Thread(target=run_http_server)
        http_server_thread.daemon = True
        http_server_thread.start()

    except Exception as e:
        logger.error(f"Error during on_ready: {e}")
        sys.exit(1)  # Exit to restart


@bot.command()
async def hello(ctx):
    await ctx.send("Hello! I'm your friendly IDL Discord bot.")


def start_bot():
    try:
        # Replace with your bot's token
        bot.run(
            "MTMyOTc2OTc0ODQ4NDk4NDg0Mw.GI5BSn.YeQTkk0qtJ6Y__KXSBD5AHI8Ym1KDeL4TickvQ"
        )
    except Exception as e:
        logger.error(f"Bot failed: {e}")
        sys.exit(1)  # Exit to restart


if __name__ == "__main__":
    # Use event loop for Python 3.6+
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_bot())
