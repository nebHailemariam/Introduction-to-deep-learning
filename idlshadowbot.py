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
        ["Mon, Feb 3", "Wed, Feb 5", "Mon, Apr 21", "Wed, Apr 23"],
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
            "Mon, Jan 27",
            "Wed, Jan 29",
            "Wed, Feb 24",
            "Mon, Mar 24",
            "Mon, Apr 21",
            "Wed, Apr 23",
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
        ["Wed, Feb 26", "Mon, Mar 17", "Mon, Mar 31", "Wed, Apr 9"],
    ],
}

import pytz

# EST timezone using pytz
est_tz = pytz.timezone("US/Eastern")


def get_est_now():
    """Get the current time in EST."""
    return datetime.datetime.now(est_tz)


def is_responsible_for_day(date_str, target_date):
    """Check if a TA is responsible for a specific date."""
    date_str_with_year = date_str + ", 2025"  # Add year to the date string
    date_obj = datetime.datetime.strptime(date_str_with_year, "%a, %b %d, %Y")
    return date_obj.date() == target_date  # Compare exact dates


def is_responsible_for_week(date_str, start_date, end_date):
    """Check if a TA has responsibilities within a given week."""
    date_str_with_year = date_str + ", 2025"
    date_obj = datetime.datetime.strptime(date_str_with_year, "%a, %b %d, %Y")
    return start_date <= date_obj.date() <= end_date


@tasks.loop(hours=24)
async def send_reminder():
    """Daily reminder task to notify TAs."""
    current_time = get_est_now()
    current_day = current_time.weekday()  # 0 = Monday, 6 = Sunday
    current_date = current_time.date()
    next_day = current_date + datetime.timedelta(days=1)  # Calculate next day's date
    next_day_str = next_day.strftime("%a, %b %d")  # Format next day's date as string
    channel = bot.get_channel(1306667280364732468)  # Replace with actual channel ID

    if channel is None:
        logger.error("Channel not found!")
        return

    if current_day == 5:  # Saturday
        # Notify about all responsibilities for the upcoming week
        start_of_week = current_date
        end_of_week = start_of_week + datetime.timedelta(days=4)  # Friday
        responsible_users = {}

        for user, info in ta_responsibilities.items():
            user_id, days = info
            for responsibility_date in days:
                if is_responsible_for_week(
                    responsibility_date, start_of_week, end_of_week
                ):
                    if user_id not in responsible_users:
                        responsible_users[user_id] = []
                    responsible_users[user_id].append(responsibility_date)

        if responsible_users:
            reminder_message = "Reminder: TAs with responsibilities next week:\n"
            for user_id, dates in responsible_users.items():
                formatted_dates = ", ".join(dates)
                reminder_message += f"<@{user_id}>: {formatted_dates}\n"
            await channel.send(reminder_message)
        else:
            await channel.send("No responsibilities assigned for next week.")

    else:
        # Notify about responsibilities for the next day
        responsible_users = []

        for user, info in ta_responsibilities.items():
            user_id, days = info
            for responsibility_date in days:
                if responsibility_date == next_day_str:
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


# Bot startup event
@bot.event
async def on_ready():
    try:
        logger.info(f"Logged in as {bot.user}.")
        send_reminder.start()  # Start the reminder loop
    except Exception as e:
        logger.error(f"Error during on_ready: {e}")


# Replace with your bot token
def start_bot():
    try:
        bot.run("TOKEN")  # Replace with your bot token
    except Exception as e:
        logger.error(f"Bot failed: {e}")


if __name__ == "__main__":
    start_bot()
