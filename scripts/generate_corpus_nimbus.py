"""Generates the "nimbus" bundled corpus and label set under
data/corpora/nimbus/ — a second bundled, hand-labeled corpus in a distinct
topic/vocabulary domain (a fictional smart-home hub product) from
generate_corpus.py's "lighthouse" (SaaS project management), mirroring its
4-doc-type mix (prose guides, code/API reference, tabular specs, FAQ) so
the pipeline's structural/similarity features are exercised the same way
across genuinely different content.

Re-run with `uv run python scripts/generate_corpus_nimbus.py` any time the
corpus content below is edited.
"""

from __future__ import annotations

from corpus_authoring import write_corpus

CORPUS_ID = "nimbus"
DISPLAY_NAME = "Nimbus"
DESCRIPTION = (
    "Fictional smart-home hub product docs: prose guides, API/SDK "
    "reference, tabular specs, and FAQs.")

DOCUMENTS: list[dict] = [
    {
        "doc_id": "nimbus-guide-getting-started",
        "title": "Getting Started with Nimbus",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Setting up your hub",
             "Plug in the Nimbus Hub and open the Nimbus app to begin "
             "setup. The app finds the hub over your local Wi-Fi network "
             "automatically; if it isn't found within a minute, hold the "
             "hub's reset button for five seconds to restart discovery."),
            ("c1", "Adding your first device",
             "From the app's Devices tab, tap Add Device and select the "
             "device type. Most Zigbee and Z-Wave devices pair "
             "automatically once put into pairing mode; Wi-Fi devices "
             "require entering your home network's password once."),
            ("c2", "Creating rooms",
             "Rooms group devices for easier control. Create a room from "
             "Settings > Rooms, then assign each device to a room from "
             "its device detail page. A device can belong to only one "
             "room at a time."),
            ("c3", "Notifications",
             "The Nimbus app sends push notifications for door/window "
             "sensor triggers and low device battery by default. You can "
             "turn off any notification category individually from "
             "Settings > Notifications."),
            ("c4", "Mobile app",
             "The Nimbus mobile app is available for iPhone and Android "
             "and mirrors every feature in the web dashboard, including "
             "live camera viewing and remote automation control."),
        ],
    },
    {
        "doc_id": "nimbus-guide-automations",
        "title": "Building Automations",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Creating a routine",
             "A routine runs a set of actions when its trigger fires. "
             "Create one from the Automations tab by choosing a trigger "
             "device or time, then adding one or more actions such as "
             "turning on a light or locking a door."),
            ("c1", "Triggers and conditions",
             "A routine's trigger starts the evaluation; optional "
             "conditions (like time of day or whether anyone is home) "
             "must all be true before the actions run. A routine with no "
             "conditions runs every time its trigger fires."),
            ("c2", "Scenes",
             "A scene is a saved set of device states, such as \"Movie "
             "Night\" dimming the lights and closing the blinds. Scenes "
             "can be triggered manually from the app or included as an "
             "action inside a routine."),
            ("c3", "Automation limits",
             "Each Nimbus account can have up to 50 active routines. "
             "Routines that reference a deleted device are automatically "
             "disabled rather than deleted, so they can be repointed at a "
             "replacement device later."),
        ],
    },
    {
        "doc_id": "nimbus-guide-sharing",
        "title": "Household Roles and Sharing",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Role overview",
             "Nimbus has four household roles: Owner, Admin, Member, and "
             "Guest. Roles are assigned per household and control what a "
             "person can see and control."),
            ("c1", "Owner permissions",
             "The Owner is the only role that can change billing details "
             "or permanently delete the household. Ownership can be "
             "transferred to an Admin from Household Settings."),
            ("c2", "Admin permissions",
             "Admins can invite or remove household members, add or "
             "remove devices, and edit any automation. Admins cannot "
             "change billing or delete the household."),
            ("c3", "Member permissions",
             "Members can control any device in the household and create "
             "their own automations. Members cannot invite new people or "
             "change another member's role."),
            ("c4", "Guest permissions",
             "Guests get temporary, time-limited control of only the "
             "devices an Admin explicitly shares with them, such as a "
             "front door lock for a house sitter. A Guest cannot view "
             "cameras unless a camera is explicitly shared."),
        ],
    },
    {
        "doc_id": "nimbus-guide-security",
        "title": "Nimbus Security Practices",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Data encryption",
             "All device data is encrypted at rest using AES-256 and in "
             "transit using TLS 1.3, both between the hub and your "
             "devices and between the hub and the Nimbus cloud."),
            ("c1", "Two-factor authentication",
             "Two-factor authentication using an authenticator app is "
             "available to every account, and an Owner can require it for "
             "every household member from Security Settings."),
            ("c2", "Firmware updates",
             "The hub checks for firmware updates nightly and installs "
             "security patches automatically. Feature updates are "
             "optional and can be deferred from Settings > Firmware."),
            ("c3", "Local vs. cloud processing",
             "Motion detection and routine evaluation run locally on the "
             "hub, so most automations keep working during an internet "
             "outage. Voice assistant integration and remote access "
             "require the cloud connection."),
        ],
    },
    {
        "doc_id": "nimbus-api-auth",
        "title": "Authentication API Reference",
        "doc_type": "code",
        "chunks": [
            ("c0", "Overview",
             "The Nimbus API supports OAuth access tokens for "
             "user-facing apps and long-lived API keys for "
             "server-to-server integrations.", "prose"),
            ("c1", "POST /v1/oauth/token",
             'curl -X POST https://api.nimbus.example/v1/oauth/token \\\n'
             '  -d client_id=CID -d client_secret=SECRET \\\n'
             '  -d grant_type=client_credentials\n'
             '# response:\n'
             '{"access_token": "nbs_...", "expires_in": 3600, '
             '"token_type": "bearer"}'),
            ("c2", "Refreshing tokens",
             'curl -X POST https://api.nimbus.example/v1/oauth/token \\\n'
             '  -d grant_type=refresh_token \\\n'
             '  -d refresh_token=$REFRESH_TOKEN\n'
             '# returns a new access_token without requiring the user to '
             'log in again'),
            ("c3", "Revoking a token",
             'curl -X DELETE https://api.nimbus.example/v1/oauth/token/'
             '{token_id} \\\n  -H "Authorization: Bearer $ACCESS_TOKEN"\n'
             '# returns 204 No Content on success'),
        ],
    },
    {
        "doc_id": "nimbus-api-devices",
        "title": "Devices API Reference",
        "doc_type": "code",
        "chunks": [
            ("c0", "Overview",
             "The devices resource represents a single paired device. "
             "Every device belongs to exactly one household and has a "
             "type, room assignment, and current state.", "prose"),
            ("c1", "GET /v1/devices/{device_id}",
             '{\n  "id": "dev_7f2a",\n  "type": "smart_lock",\n'
             '  "room_id": "rm_1c88",\n  "state": "locked",\n'
             '  "battery_pct": 82\n}'),
            ("c2", "PATCH /v1/devices/{device_id}",
             'curl -X PATCH https://api.nimbus.example/v1/devices/dev_7f2a '
             '\\\n  -d state=unlocked\n'
             '# any subset of settable device fields may be included'),
            ("c3", "Rate limiting headers",
             'Every API response includes:\n'
             'X-RateLimit-Limit: 300\n'
             'X-RateLimit-Remaining: 271\n'
             'X-RateLimit-Reset: 1735689600'),
        ],
    },
    {
        "doc_id": "nimbus-api-automations",
        "title": "Automations API Reference",
        "doc_type": "code",
        "chunks": [
            ("c0", "Overview",
             "Automations created in the app are also manageable through "
             "the API. Supported trigger types include device_state, "
             "time_of_day, and geofence.", "prose"),
            ("c1", "POST /v1/automations",
             'curl -X POST https://api.nimbus.example/v1/automations \\\n'
             '  -d name="Porch light at dusk" \\\n'
             '  -d trigger_type=time_of_day -d trigger_value=sunset'),
            ("c2", "Listing automations",
             'curl https://api.nimbus.example/v1/automations \\\n'
             '  -H "Authorization: Bearer $ACCESS_TOKEN"\n'
             '# returns every automation for the authenticated household'),
            ("c3", "Deleting an automation",
             'curl -X DELETE https://api.nimbus.example/v1/automations/'
             'aut_55b2\n'
             '# any routine referencing a deleted device is disabled, not '
             'deleted, per the automation limits guide'),
        ],
    },
    {
        "doc_id": "nimbus-sdk-python",
        "title": "Python SDK Quickstart",
        "doc_type": "code",
        "chunks": [
            ("c0", "Installation", "pip install nimbus-sdk"),
            ("c1", "Client initialization",
             'from nimbus import Nimbus\n'
             'client = Nimbus(api_key="nbs_...")'),
            ("c2", "Reading a device's state",
             'lock = client.devices.get("dev_7f2a")\n'
             'print(lock.state)'),
            ("c3", "Listing devices with a filter",
             'locks = client.devices.list(\n'
             '    type="smart_lock",\n    room_id="rm_1c88",\n)'),
            ("c4", "Error handling",
             'from nimbus import NimbusAPIError\n'
             'try:\n    client.devices.patch("bad_id", state="unlocked")\n'
             'except NimbusAPIError as e:\n'
             '    print(e.status_code, e.message)'),
        ],
    },
    {
        "doc_id": "nimbus-pricing-table",
        "title": "Pricing and Plans",
        "doc_type": "table",
        "chunks": [
            ("c0", "Plan comparison table",
             "| | Free | Plus | Pro |\n"
             "|---|---|---|---|\n"
             "| Price per month | $0 | $6 | Custom |\n"
             "| Max devices | 10 | 50 | Unlimited |\n"
             "| Cloud recording | No | 7 days | 30 days |\n"
             "| Geofencing | No | Yes | Yes |\n"
             "| API access | No | No | Yes |"),
            ("c1", "Storage limits table",
             "| Plan | Video storage |\n|---|---|\n"
             "| Free | None |\n| Plus | 10 GB |\n"
             "| Pro | 200 GB |"),
            ("c2", "API rate limits table",
             "| Plan | Requests per minute |\n|---|---|\n"
             "| Free | 0 |\n| Plus | 0 |\n"
             "| Pro | 300 |"),
        ],
    },
    {
        "doc_id": "nimbus-device-compatibility",
        "title": "Device Compatibility Reference",
        "doc_type": "table",
        "chunks": [
            ("c0", "Supported protocols table",
             "| Protocol | Pairing method |\n|---|---|\n"
             "| Zigbee | Automatic once in pairing mode |\n"
             "| Z-Wave | Automatic once in pairing mode |\n"
             "| Wi-Fi | Requires network password |\n"
             "| Bluetooth | Requires phone in range during setup |"),
            ("c1", "Supported device types table",
             "| Type | Examples |\n|---|---|\n"
             "| Smart lock | Deadbolt, padlock |\n"
             "| Sensor | Door/window, motion, water leak |\n"
             "| Light | Bulb, dimmer switch |\n"
             "| Camera | Indoor, outdoor, doorbell |"),
        ],
    },
    {
        "doc_id": "nimbus-system-requirements",
        "title": "System Requirements",
        "doc_type": "table",
        "chunks": [
            ("c0", "Hub requirements table",
             "| Requirement | Minimum |\n|---|---|\n"
             "| Wi-Fi | 2.4 GHz, WPA2 |\n"
             "| Power | USB-C, 5W |\n"
             "| Internet | 5 Mbps for cloud features |"),
            ("c1", "Mobile app requirements table",
             "| Platform | Minimum version |\n|---|---|\n"
             "| iOS | 15 |\n| Android | 10 |"),
            ("c2", "Browser support table",
             "| Browser | Minimum version |\n|---|---|\n"
             "| Chrome | 100 |\n| Firefox | 100 |\n"
             "| Safari | 15 |\n| Edge | 100 |"),
        ],
    },
    {
        "doc_id": "nimbus-status-codes",
        "title": "HTTP Status Code Reference",
        "doc_type": "table",
        "chunks": [
            ("c0", "Success codes table",
             "| Code | Meaning |\n|---|---|\n"
             "| 200 | Request succeeded |\n"
             "| 201 | Resource created |\n"
             "| 204 | Succeeded with no response body, e.g. after a "
             "delete |"),
            ("c1", "Client error codes table",
             "| Code | Meaning |\n|---|---|\n"
             "| 400 | Malformed request |\n"
             "| 401 | Missing or invalid credentials |\n"
             "| 403 | Authenticated but not permitted |\n"
             "| 404 | Resource not found |\n"
             "| 429 | Too many requests, rate limited |"),
            ("c2", "Server error codes table",
             "| Code | Meaning |\n|---|---|\n"
             "| 500 | Unexpected server error |\n"
             "| 503 | Service temporarily unavailable |"),
        ],
    },
    {
        "doc_id": "nimbus-faq-account",
        "title": "Account and Access FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "How do I reset my password?",
             "Tap Forgot password on the login screen and follow the "
             "emailed reset link. No household admin needs to be "
             "involved."),
            ("c1", "Can I merge two accounts?",
             "Account merging is not currently automated. Contact "
             "support with both account email addresses and a support "
             "engineer will merge them manually."),
            ("c2", "How do I transfer household ownership?",
             "The current Owner can go to Household Settings > Transfer "
             "Ownership and select any existing Admin as the new Owner."),
            ("c3", "What role does the previous owner get after a "
             "transfer?",
             "The previous Owner is automatically demoted to Admin. They "
             "keep full device and automation access except for billing "
             "changes, which are locked for 24 hours as a safety "
             "cooldown."),
            ("c4", "Can Guests be promoted to Members?",
             "Yes. Any Admin or Owner can promote a Guest to a full "
             "Member from the household members list."),
        ],
    },
    {
        "doc_id": "nimbus-faq-billing",
        "title": "Billing FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Can I change plans mid-cycle?",
             "Yes. Upgrades take effect immediately and you are charged "
             "a prorated amount for the rest of the billing cycle. "
             "Downgrades take effect at your next renewal date."),
            ("c1", "What payment methods are accepted?",
             "Credit card is accepted on the Free and Plus plans. Pro "
             "plans billed annually may also pay by ACH bank transfer "
             "against an invoice."),
            ("c2", "How do I get a copy of an invoice?",
             "Open the Billing page and tap the Invoices tab. Every past "
             "invoice can be downloaded as a PDF from there."),
            ("c3", "What happens if a payment fails?",
             "There is a 7 day grace period after a failed payment "
             "during which the hub keeps working normally. If payment "
             "still has not succeeded after 14 days, the household is "
             "automatically downgraded to the Free plan."),
            ("c4", "Is footage lost when a household is auto-downgraded "
             "to Free?",
             "No footage already recorded is deleted when a household is "
             "downgraded, but the Free plan retains no new cloud "
             "recordings going forward until you upgrade again."),
            ("c5", "Can I get a refund?",
             "Refunds are only available for charges made within the "
             "last 14 days, and are prorated based on the unused portion "
             "of the billing period."),
        ],
    },
    {
        "doc_id": "nimbus-faq-technical",
        "title": "Technical FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Does Nimbus work without an internet connection?",
             "Locally-evaluated routines and manual device control over "
             "the local network keep working during an outage. Remote "
             "access, cloud recording, and voice assistants require the "
             "cloud connection."),
            ("c1", "Why am I hitting rate limit errors?",
             "You are exceeding your plan's allowed API requests per "
             "minute; see the API rate limits table in the pricing "
             "documentation for your plan's exact limit."),
            ("c2", "How many active routines can I have?",
             "Each account can have up to 50 active routines, per the "
             "automation limits section of the automations guide."),
            ("c3", "Can I export all of my data?",
             "Yes, from Settings > Data Export you can download a CSV "
             "export containing your devices, rooms, and automations."),
            ("c4", "Is there a public status page?",
             "status.nimbus.example shows current uptime and a history "
             "of past incidents."),
        ],
    },
    {
        "doc_id": "nimbus-faq-security",
        "title": "Security FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Is two-factor authentication mandatory?",
             "Two-factor authentication is optional per user by "
             "default, unless a household Owner turns on org-wide "
             "enforcement in Security Settings."),
            ("c1", "Where is my data hosted?",
             "Account and device metadata is hosted on AWS in the "
             "us-east-1 region by default. EU data residency is "
             "available as an add-on for Pro plans."),
            ("c2", "Do you have a bug bounty program?",
             "Yes, Nimbus runs a bug bounty program hosted on "
             "HackerOne. Its scope covers app.nimbus.example and the "
             "hub's local API only."),
            ("c3", "How do I report a security vulnerability?",
             "Email security@nimbus.example. A PGP key for encrypting "
             "sensitive reports is published on the security page."),
        ],
    },
]

# (question_id, question_text, [(doc_id, chunk_suffix), ...])
QUESTIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("n01", "How do I get the hub to find my devices during setup?",
     [("nimbus-guide-getting-started", "c0")]),
    ("n02", "How do I pair a Wi-Fi device with Nimbus?",
     [("nimbus-guide-getting-started", "c1")]),
    ("n03", "Can a device belong to more than one room?",
     [("nimbus-guide-getting-started", "c2")]),
    ("n04", "How do I turn off low battery notifications?",
     [("nimbus-guide-getting-started", "c3")]),
    ("n05", "Does the Nimbus app support viewing cameras live?",
     [("nimbus-guide-getting-started", "c4")]),
    ("n06", "What happens to a routine if it has no conditions?",
     [("nimbus-guide-automations", "c1")]),
    ("n07", "What is a scene in Nimbus?",
     [("nimbus-guide-automations", "c2")]),
    ("n08", "How many automations can one account have?",
     [("nimbus-guide-automations", "c3")]),
    ("n09", "What can a Guest control in someone else's household?",
     [("nimbus-guide-sharing", "c4")]),
    ("n10", "Who can invite new members to a household?",
     [("nimbus-guide-sharing", "c2")]),
    ("n11", "Who is allowed to delete an entire household?",
     [("nimbus-guide-sharing", "c1")]),
    ("n12", "Is data encrypted between the hub and the cloud?",
     [("nimbus-guide-security", "c0")]),
    ("n13", "Do automations still run if the internet goes down?",
     [("nimbus-guide-security", "c3")]),
    ("n14", "What fields are returned when I fetch a single device?",
     [("nimbus-api-devices", "c1")]),
    ("n15", "How do I unlock a smart lock through the API?",
     [("nimbus-api-devices", "c2")]),
    ("n16", "What response headers show my remaining API requests?",
     [("nimbus-api-devices", "c3")]),
    ("n17", "How do I get an access token to call the Nimbus API?",
     [("nimbus-api-auth", "c1")]),
    ("n18", "How can I refresh an expired access token without logging "
     "in again?", [("nimbus-api-auth", "c2")]),
    ("n19", "What trigger types are supported for API-created "
     "automations?", [("nimbus-api-automations", "c0")]),
    ("n20", "How do I create an automation that turns on a light at "
     "sunset via the API?", [("nimbus-api-automations", "c1")]),
    ("n21", "What happens to an automation's referenced device if I "
     "delete it through the API?", [("nimbus-api-automations", "c3")]),
    ("n22", "How do I install the Nimbus Python SDK?",
     [("nimbus-sdk-python", "c0")]),
    ("n23", "How do I read a device's current state using the Python "
     "SDK?", [("nimbus-sdk-python", "c2")]),
    ("n24", "How much cloud video storage does the Plus plan include?",
     [("nimbus-pricing-table", "c1")]),
    ("n25", "Does the Free plan include geofencing?",
     [("nimbus-pricing-table", "c0")]),
    ("n26", "What API rate limit applies to the Pro plan?",
     [("nimbus-pricing-table", "c2")]),
    ("n27", "How does a Zigbee device get paired with the hub?",
     [("nimbus-device-compatibility", "c0")]),
    ("n28", "Is a video doorbell a supported device type?",
     [("nimbus-device-compatibility", "c1")]),
    ("n29", "What Wi-Fi frequency does the hub require?",
     [("nimbus-system-requirements", "c0")]),
    ("n30", "What does a 429 status code mean?",
     [("nimbus-status-codes", "c1")]),
    ("n31", "What does a 204 response mean when I delete an "
     "automation?", [("nimbus-status-codes", "c0")]),
    ("n32", "Can I change my Nimbus plan in the middle of a billing "
     "cycle?", [("nimbus-faq-billing", "c0")]),
    ("n33", "What payment methods does Nimbus accept for annual Pro "
     "billing?", [("nimbus-faq-billing", "c1")]),
    ("n34", "Where can I download a copy of my invoice?",
     [("nimbus-faq-billing", "c2")]),
    ("n35", "If my payment fails and my household gets downgraded to "
     "Free, do I lose footage I already recorded?",
     [("nimbus-faq-billing", "c3"), ("nimbus-faq-billing", "c4")]),
    ("n36", "Can I get my money back after being charged?",
     [("nimbus-faq-billing", "c5")]),
    ("n37", "How do I reset a forgotten password?",
     [("nimbus-faq-account", "c0")]),
    ("n38", "After I transfer household ownership to someone else, what "
     "role do I end up with?",
     [("nimbus-faq-account", "c2"), ("nimbus-faq-account", "c3")]),
    ("n39", "Can an Admin promote a Guest to a full Member?",
     [("nimbus-faq-account", "c4")]),
    ("n40", "Does Nimbus keep working at all without an internet "
     "connection?", [("nimbus-faq-technical", "c0")]),
    ("n41", "Why do I keep getting rate limited when calling the API?",
     [("nimbus-faq-technical", "c1")]),
    ("n42", "How many active routines am I allowed to have, and where "
     "is that documented?",
     [("nimbus-faq-technical", "c2"), ("nimbus-guide-automations", "c3")]),
    ("n43", "How do I export all of my Nimbus data?",
     [("nimbus-faq-technical", "c3")]),
    ("n44", "Is two-factor authentication required for everyone in my "
     "household?",
     [("nimbus-faq-security", "c0"), ("nimbus-guide-security", "c1")]),
    ("n45", "Where is my account data hosted by default?",
     [("nimbus-faq-security", "c1")]),
    ("n46", "How do I take part in Nimbus's bug bounty and report a "
     "vulnerability I found?",
     [("nimbus-faq-security", "c2"), ("nimbus-faq-security", "c3")]),
]


def main() -> None:
    write_corpus(CORPUS_ID, DISPLAY_NAME, DESCRIPTION, DOCUMENTS, QUESTIONS)


if __name__ == "__main__":
    main()
