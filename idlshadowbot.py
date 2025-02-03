import discord
from discord.ext import commands, tasks
import datetime
import pytz
import logging
import asyncio

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize bot intents and bot object
intents = discord.Intents.default()
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Timezone for Pittsburgh (EST)
est_tz = pytz.timezone("America/New_York")

ta_responsibilities = {
    "Kateryna Shapovalenko": [
        1134188563236409404,
        [
            "Mon, Feb 3",
            "Wed, Feb 5",
            "Mon, Apr 14",
            "Wed, Apr 16",
        ],
    ],
    "Miya Sylvester": [
        1045426392248094780,
        [
            "Mon, Apr 21",
            "Wed, Apr 23",
            "Mon, Mar 31",
            "Wed, Apr 9",
        ],
    ],
    "Alexander Moker": [
        225488327036370944,
        ["Mon, Feb 24", "Wed, Feb 26"],
    ],
    "Purusottam Samal": [
        582931106605563904,
        ["Mon, Jan 27", "Wed, Jan 29"],
    ],
    "Shravanth Srinivas": [
        847319475040157768,
        ["Mon, Jan 27", "Wed, Jan 29"],
    ],
    "Yuzhou Wang": [
        747430516810317864,
        ["Mon, Feb 3", "Mon, Apr 7", "Wed, Apr 9"],
    ],
    "Massa Baali": [
        1191394865750216807,
        ["Mon, Feb 17", "Wed, Feb 19", "Mon, Mar 17", "Wed, Mar 19"],
    ],
    "Vedant Singh": [
        1181754582414675989,
        ["Mon, Feb 10", "Wed, Feb 12", "Mon, Mar 10", "Wed, Mar 12"],
    ],
    "Sadrishya Agrawal": [
        617383770500235276,
        ["Mon, Mar 31", "Wed, Apr 9", "Mon, Apr 14", "Wed, Apr 16"],
    ],
    "Michael Kireeff": [
        1056406329092218890,
        ["Mon, Feb 10", "Wed, Feb 12", "Wed, Mar 19", "Wed, Mar 26"],
    ],
    "Vishan Oberoi": [
        756414206349410314,
        ["Mon, Mar 24", "Wed, Mar 26", "Wed, Apr 2", "Mon, Apr 7"],
    ],
    "Ishita Gupta": [
        768524190097342466,
        ["Mon, Jan 27", "Wed, Jan 29"],
    ],
    "Shrey Jain": [
        1280280367940632667,
        ["Mon, Mar 24", "Wed, Mar 26", "Wed, Apr 2", "Mon, Apr 7"],
    ],
    "Floris Nzabakira": [
        1308581724107571351,
        ["Mon, Feb 24", "Wed, Feb 26", "Mon, Mar 10", "Wed, Mar 12"],
    ],
    "Ahmed Issah": [
        1206316006041976893,
        ["Mon, Mar 31", "Wed, Apr 2", "Mon, Apr 14", "Wed, Apr 16"],
    ],
    "Shubham Singh (Sigma)": [
        1308615486543499315,
        ["Mon, Jan 27", "Wed, Jan 29"],
    ],
    "Tanghang Elvis Tata": [
        1311026963040960563,
        [
            # "Mon, Jan 27",
            # "Wed, Jan 29",
            # "Wed, Feb 24",
            # "Mon, Mar 24",
            # "Mon, Apr 21",
            # "Wed, Apr 23",
        ],
    ],
    "John Liu": [
        1311369823468388363,
        ["Mon, Jan 27", "Wed, Jan 29"],
    ],
    "Eman Ensar": [
        744141074779471919,
        ["Mon, Feb 17", "Wed, Feb 19", "Mon, Mar 10", "Wed, Mar 12"],
    ],
    "Damilare Olatunji": [
        1259328537266094162,
        ["Mon, Feb 10", "Wed, Feb 12", "Mon, Feb 17", "Wed, Feb 19"],
    ],
    "Brian Ebiyau": [
        947067276732014613,
        ["Mon, Feb 24", "Wed, Feb 26", "Mon, Apr 21", "Wed, Apr 23"],
    ],
    "Peter Wauyo": [
        1247696351425466498,
        [
            "Wed, Feb 26",
            "Mon, Mar 17",
            "Mon, Feb 03",
            "Wed, Feb 5",
        ],
    ],
}


def get_est_now():
    """Get the current time in Pittsburgh (EST)."""
    return datetime.datetime.now(est_tz)


def get_next_run_time():
    """Get the next scheduled time at 10 AM EST."""
    now = get_est_now()
    next_run = now.replace(hour=10, minute=0, second=0, microsecond=0)
    if now >= next_run:  # If it's past 10 AM today, schedule for tomorrow
        next_run += datetime.timedelta(days=1)
    return next_run


def get_seconds_until_next_run():
    """Get seconds until the next scheduled 10 AM EST run."""
    now = get_est_now()
    next_run = get_next_run_time()
    return (next_run - now).total_seconds()


def normalize_date(date_str):
    """Normalize dates to handle both 'Feb 3' and 'Feb 03' cases."""
    try:
        return datetime.datetime.strptime(
            date_str + ", 2025", "%a, %b %d, %Y"
        ).strftime("%a, %b %-d")
    except ValueError:
        return datetime.datetime.strptime(
            date_str + ", 2025", "%a, %b %d, %Y"
        ).strftime("%a, %b %-d")


@tasks.loop(hours=24)
async def send_reminder():
    """Send daily reminders at 10 AM EST."""
    await bot.wait_until_ready()
    current_time = get_est_now()
    current_day = current_time.weekday()  # 0 = Monday, 6 = Sunday
    current_date = current_time.date()
    next_day = current_date + datetime.timedelta(days=1)
    next_day_str = next_day.strftime("%a, %b %-d")
    channel = bot.get_channel(1306667280364732468)  # Replace with actual channel ID

    if channel is None:
        logger.error("Channel not found!")
        return

    responsible_users = []

    for user, info in ta_responsibilities.items():
        user_id, days = info
        for responsibility_date in days:
            if normalize_date(responsibility_date) == next_day_str:
                responsible_users.append(f"<@{user_id}>")

    if responsible_users:
        reminder_message = (
            "Reminder: "
            + " ".join(responsible_users)
            + ", you have a Shadow responsibility tomorrow!"
        )
        await channel.send(reminder_message)
    else:
        await channel.send(f"No responsibilities tomorrow ({next_day_str}).")


@bot.event
async def on_ready():
    """Run the bot and send an initial reminder immediately."""
    logger.info(f"Logged in as {bot.user}.")

    # Send an immediate notification
    channel = bot.get_channel(1290436547094777939)
    if channel:
        await channel.send("Bot is online! Sending an initial reminder...")
        await send_reminder()

    # Schedule task to run at 10 AM EST
    seconds_until_next_run = get_seconds_until_next_run()
    logger.info(f"Next reminder scheduled in {seconds_until_next_run} seconds.")
    await asyncio.sleep(seconds_until_next_run)  # Wait until 10 AM EST
    send_reminder.start()


def start_bot():
    try:
        bot.run("Token")  # Replace with actual bot token
    except Exception as e:
        logger.error(f"Bot failed: {e}")


if __name__ == "__main__":
    start_bot()
