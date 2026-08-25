"""Generates the "lighthouse" bundled corpus and label set under
data/corpora/lighthouse/.

Re-run with `uv run python scripts/generate_corpus.py` any time the corpus
content below is edited. The emitted files (data/corpora/lighthouse/...)
are checked into the repo as the reproducible source of truth; this script
is the editable source for that content. See generate_corpus_nimbus.py for
the second bundled corpus, and corpus_authoring.py for the shared writer.
"""

from __future__ import annotations

from corpus_authoring import write_corpus

CORPUS_ID = "lighthouse"
DISPLAY_NAME = "Lighthouse"
DESCRIPTION = (
    "Fictional SaaS project-management product docs: prose guides, "
    "API/SDK reference, tabular specs, and FAQs.")

# Each document: doc_id, title, doc_type (default source_type for its chunks),
# and chunks as (chunk_id_suffix, heading, text) or (chunk_id_suffix, heading,
# text, source_type_override).
DOCUMENTS: list[dict] = [
    {
        "doc_id": "guide-getting-started",
        "title": "Getting Started with Lighthouse",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Creating your workspace",
             "When you sign up for Lighthouse, you create a workspace first. "
             "From the workspace settings you can invite teammates by email; "
             "each invited teammate receives a link to join and choose a "
             "password. There is no limit on the number of pending invites "
             "you can send at once."),
            ("c1", "Creating your first project",
             "Inside a workspace, click New Project to create a board. Every "
             "new project starts with three default columns: To Do, In "
             "Progress, and Done. You can rename or add columns later from "
             "the board settings menu."),
            ("c2", "Adding tasks",
             "To add a task, click the plus icon at the bottom of a column. "
             "Each task can be assigned to a teammate and given a due date. "
             "Tasks without an assignee show up in the Unassigned filter so "
             "nothing falls through the cracks."),
            ("c3", "Notifications",
             "Lighthouse sends notifications both by email and inside the "
             "app. You can turn off email notifications entirely from "
             "Settings > Notifications while keeping in-app alerts on, or "
             "the reverse."),
            ("c4", "Mobile app",
             "Lighthouse has a native mobile app available for iPhone and "
             "Android. The mobile app supports viewing and editing tasks, "
             "and push notifications for assignments and mentions."),
        ],
    },
    {
        "doc_id": "guide-permissions",
        "title": "Permissions and Roles Guide",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Role overview",
             "Lighthouse has four roles: Owner, Admin, Member, and Guest. "
             "Roles are assigned per workspace and control what a person can "
             "see and change."),
            ("c1", "Owner permissions",
             "The Owner is the only role that can change billing details or "
             "permanently delete the entire workspace. Ownership can be "
             "transferred to an Admin from the workspace settings."),
            ("c2", "Admin permissions",
             "Admins can manage workspace members, create and archive "
             "projects, and connect or disconnect integrations. Admins "
             "cannot change billing or delete the workspace."),
            ("c3", "Member permissions",
             "Members can create and edit tasks in any project they have "
             "been added to. Members cannot invite new people to the "
             "workspace or change another member's role."),
            ("c4", "Guest permissions",
             "Guests have view-only access to the specific projects they "
             "were invited to. A Guest cannot invite other people, cannot "
             "edit tasks, and cannot see any project they were not "
             "explicitly added to."),
        ],
    },
    {
        "doc_id": "guide-integrations",
        "title": "Connecting Third-Party Integrations",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Slack integration",
             "Connect Lighthouse to Slack from Settings > Integrations to "
             "post task updates directly into a chosen channel whenever a "
             "task changes status."),
            ("c1", "GitHub integration",
             "Linking a GitHub repository lets you reference a task from a "
             "commit message or pull request title using its task ID; "
             "merging the linked pull request automatically marks the task "
             "done."),
            ("c2", "Google Calendar integration",
             "Enabling the Google Calendar integration syncs every task "
             "with a due date into a dedicated Lighthouse calendar that "
             "updates automatically as dates change."),
            ("c3", "Zapier integration",
             "Lighthouse connects to more than five thousand other "
             "applications through Zapier, letting you build custom "
             "automations without writing any code."),
        ],
    },
    {
        "doc_id": "guide-security",
        "title": "Security Practices Overview",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Data encryption",
             "All customer data is encrypted at rest using AES-256 and in "
             "transit using TLS 1.3. Encryption keys are rotated on a "
             "regular schedule managed by the infrastructure team."),
            ("c1", "Single sign-on",
             "Enterprise plan workspaces can enable single sign-on through "
             "any identity provider that supports SAML 2.0, such as Okta or "
             "Azure AD."),
            ("c2", "Two-factor authentication",
             "Two-factor authentication using an authenticator app is "
             "available to every user, and Admins can enforce it as "
             "mandatory for the entire workspace from the security "
             "settings."),
            ("c3", "Audit logs",
             "Enterprise plan workspaces retain a full audit log of "
             "membership and permission changes for one year, exportable at "
             "any time as a CSV file."),
        ],
    },
    {
        "doc_id": "api-auth",
        "title": "Authentication API Reference",
        "doc_type": "code",
        "chunks": [
            ("c0", "Overview",
             "The Lighthouse API supports two authentication methods: "
             "short-lived OAuth access tokens for user-facing apps, and "
             "long-lived API keys for server-to-server integrations.",
             "prose"),
            ("c1", "POST /v1/auth/token",
             'curl -X POST https://api.lighthouse.example/v1/auth/token \\\n'
             '  -d client_id=CID -d client_secret=SECRET \\\n'
             '  -d grant_type=client_credentials\n'
             '# response:\n'
             '{"access_token": "ltk_...", "expires_in": 3600, '
             '"token_type": "bearer"}'),
            ("c2", "Refreshing tokens",
             'curl -X POST https://api.lighthouse.example/v1/auth/token \\\n'
             '  -d grant_type=refresh_token \\\n'
             '  -d refresh_token=$REFRESH_TOKEN\n'
             '# returns a new access_token without requiring the user to '
             'log in again'),
            ("c3", "Revoking a token",
             'curl -X DELETE https://api.lighthouse.example/v1/auth/token/'
             '{token_id} \\\n  -H "Authorization: Bearer $ACCESS_TOKEN"\n'
             '# returns 204 No Content on success'),
        ],
    },
    {
        "doc_id": "api-tasks",
        "title": "Tasks API Reference",
        "doc_type": "code",
        "chunks": [
            ("c0", "Overview",
             "The tasks resource represents a single card on a project "
             "board. Every task belongs to exactly one project and has a "
             "status, priority, and optional assignee.", "prose"),
            ("c1", "GET /v1/tasks/{task_id}",
             '{\n  "id": "tsk_9f2a",\n  "title": "Write release notes",\n'
             '  "status": "in_progress",\n  "assignee_id": "usr_1c88",\n'
             '  "due_date": "2026-09-01"\n}'),
            ("c2", "POST /v1/tasks",
             'curl -X POST https://api.lighthouse.example/v1/tasks \\\n'
             '  -d project_id=prj_44a1 -d title="Ship the release" \\\n'
             '  -d assignee_id=usr_1c88'),
            ("c3", "PATCH /v1/tasks/{task_id}",
             'curl -X PATCH https://api.lighthouse.example/v1/tasks/tsk_9f2a '
             '\\\n  -d status=done\n'
             '# any subset of task fields may be included in the body'),
            ("c4", "Rate limiting headers",
             'Every API response includes:\n'
             'X-RateLimit-Limit: 600\n'
             'X-RateLimit-Remaining: 542\n'
             'X-RateLimit-Reset: 1735689600'),
        ],
    },
    {
        "doc_id": "api-webhooks",
        "title": "Webhooks API Reference",
        "doc_type": "code",
        "chunks": [
            ("c0", "Overview",
             "Webhooks notify your server when events happen in Lighthouse. "
             "Supported event types include task.created, task.completed, "
             "and project.archived.", "prose"),
            ("c1", "Registering a webhook",
             'curl -X POST https://api.lighthouse.example/v1/webhooks \\\n'
             '  -d target_url=https://yourapp.example/hooks \\\n'
             '  -d event_types=task.completed,project.archived'),
            ("c2", "Verifying webhook signatures",
             '# Lighthouse signs each payload in the X-Lighthouse-Signature '
             'header\n'
             'expected = hmac_sha256(webhook_secret, request_body)\n'
             'if not constant_time_equal(expected, header_value):\n'
             '    reject_request()'),
            ("c3", "Retry behavior",
             "If your endpoint does not return a 2xx response, Lighthouse "
             "retries delivery up to five times using exponential backoff, "
             "giving up after 24 hours of failed attempts."),
        ],
    },
    {
        "doc_id": "sdk-python",
        "title": "Python SDK Quickstart",
        "doc_type": "code",
        "chunks": [
            ("c0", "Installation", "pip install lighthouse-sdk"),
            ("c1", "Client initialization",
             'from lighthouse import Lighthouse\n'
             'client = Lighthouse(api_key="ltk_...")'),
            ("c2", "Creating a task",
             'client.tasks.create(\n'
             '    project_id="prj_44a1",\n'
             '    title="Ship the release",\n'
             '    assignee_id="usr_1c88",\n)'),
            ("c3", "Listing tasks with a filter",
             'open_tasks = client.tasks.list(\n'
             '    status="open",\n    assignee_id="usr_1c88",\n)'),
            ("c4", "Error handling",
             'from lighthouse import LighthouseAPIError\n'
             'try:\n    client.tasks.create(project_id="bad_id", title="x")\n'
             'except LighthouseAPIError as e:\n'
             '    print(e.status_code, e.message)'),
        ],
    },
    {
        "doc_id": "pricing-table",
        "title": "Pricing and Plans",
        "doc_type": "table",
        "chunks": [
            ("c0", "Plan comparison table",
             "| | Free | Team | Enterprise |\n"
             "|---|---|---|---|\n"
             "| Price per seat | $0 | $9/mo | Custom |\n"
             "| Max projects | 3 | Unlimited | Unlimited |\n"
             "| Integrations | Slack only | All | All |\n"
             "| Single sign-on | No | No | Yes |\n"
             "| Audit logs | No | No | Yes |"),
            ("c1", "Storage limits table",
             "| Plan | Storage |\n|---|---|\n"
             "| Free | 1 GB |\n| Team | 50 GB |\n"
             "| Enterprise | Unlimited |"),
            ("c2", "API rate limits table",
             "| Plan | Requests per minute |\n|---|---|\n"
             "| Free | 60 |\n| Team | 600 |\n"
             "| Enterprise | 6000 |"),
        ],
    },
    {
        "doc_id": "system-requirements",
        "title": "System Requirements",
        "doc_type": "table",
        "chunks": [
            ("c0", "Desktop app requirements",
             "| OS | Minimum version | RAM |\n|---|---|---|\n"
             "| Windows | 10 | 4 GB |\n| macOS | 12 | 4 GB |\n"
             "| Ubuntu | 20.04 | 4 GB |"),
            ("c1", "Mobile app requirements",
             "| Platform | Minimum version |\n|---|---|\n"
             "| iOS | 15 |\n| Android | 10 |"),
            ("c2", "Browser support table",
             "| Browser | Minimum version |\n|---|---|\n"
             "| Chrome | 100 |\n| Firefox | 100 |\n"
             "| Safari | 15 |\n| Edge | 100 |"),
        ],
    },
    {
        "doc_id": "field-reference",
        "title": "Task Field Reference",
        "doc_type": "table",
        "chunks": [
            ("c0", "Core fields table",
             "| Field | Type | Description |\n|---|---|---|\n"
             "| id | string | Unique task identifier |\n"
             "| title | string | Task name |\n"
             "| status | enum | todo, in_progress, or done |\n"
             "| priority | enum | low, medium, or high |\n"
             "| due_date | date | Optional due date |"),
            ("c1", "Custom field types table",
             "| Type | Max per project |\n|---|---|\n"
             "| text | 20 |\n| number | 20 |\n"
             "| dropdown | 10 |\n| checkbox | 20 |\n| date | 10 |"),
        ],
    },
    {
        "doc_id": "status-codes",
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
        "doc_id": "faq-billing",
        "title": "Billing FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Can I change plans mid-cycle?",
             "Yes. Upgrades take effect immediately and you are charged a "
             "prorated amount for the rest of the billing cycle. Downgrades "
             "take effect at your next renewal date."),
            ("c1", "What payment methods are accepted?",
             "Credit card is accepted on the Free and Team plans. "
             "Enterprise plans billed annually may also pay by ACH bank "
             "transfer against an invoice."),
            ("c2", "How do I get a copy of an invoice?",
             "Open the Billing page and click the Invoices tab. Every past "
             "invoice can be downloaded as a PDF from there."),
            ("c3", "What happens if a payment fails?",
             "There is a 7 day grace period after a failed payment during "
             "which the workspace keeps working normally. If payment still "
             "has not succeeded after 14 days, the workspace is "
             "automatically downgraded to the Free plan."),
            ("c4", "Is data lost when a workspace is auto-downgraded to "
             "Free?",
             "No data is deleted when a workspace is downgraded. Projects "
             "beyond the Free plan's project limit simply become read-only "
             "until you either upgrade again or remove projects to fit "
             "within the limit."),
            ("c5", "Can I get a refund?",
             "Refunds are only available for charges made within the last "
             "14 days, and are prorated based on the unused portion of the "
             "billing period."),
        ],
    },
    {
        "doc_id": "faq-account",
        "title": "Account and Access FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "How do I reset my password?",
             "Click Forgot password on the login screen and follow the "
             "emailed reset link. No workspace admin needs to be involved."),
            ("c1", "Can I merge two accounts?",
             "Account merging is not currently automated. Contact support "
             "with both account email addresses and a support engineer will "
             "merge them manually."),
            ("c2", "How do I transfer workspace ownership?",
             "The current Owner can go to Workspace Settings > Transfer "
             "Ownership and select any existing Admin as the new Owner."),
            ("c3", "What role does the previous owner get after a "
             "transfer?",
             "The previous Owner is automatically demoted to Admin. They "
             "keep full management access except for billing changes, "
             "which are locked for 24 hours as a safety cooldown."),
            ("c4", "Can Guests be promoted to Members?",
             "Yes. Any Admin or Owner can promote a Guest to a full Member "
             "from the workspace members list."),
        ],
    },
    {
        "doc_id": "faq-technical",
        "title": "Technical FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Does Lighthouse support offline mode?",
             "Offline read access is available only in the mobile apps. "
             "The desktop and web apps require an active connection to "
             "load or save changes."),
            ("c1", "Why am I hitting rate limit errors?",
             "You are exceeding your plan's allowed API requests per "
             "minute; see the API rate limits table in the pricing "
             "documentation for your plan's exact limit."),
            ("c2", "How long are webhook retries attempted?",
             "A failing webhook endpoint is retried for up to 24 hours "
             "using exponential backoff before Lighthouse stops trying."),
            ("c3", "Can I export all of my data?",
             "Yes, from Settings > Data Export you can download a CSV "
             "export containing your tasks, projects, and comments."),
            ("c4", "Is there a public status page?",
             "status.lighthouse.example shows current uptime and a history "
             "of past incidents."),
        ],
    },
    {
        "doc_id": "faq-security",
        "title": "Security FAQ",
        "doc_type": "prose",
        "chunks": [
            ("c0", "Is two-factor authentication mandatory?",
             "Two-factor authentication is optional per user by default, "
             "unless a workspace Admin turns on org-wide enforcement in the "
             "security settings."),
            ("c1", "Where is customer data hosted?",
             "Customer data is hosted on AWS in the us-east-1 region by "
             "default. EU data residency is available as an add-on for "
             "Enterprise plans."),
            ("c2", "Do you have a bug bounty program?",
             "Yes, Lighthouse runs a bug bounty program hosted on "
             "HackerOne. Its scope covers app.lighthouse.example only."),
            ("c3", "How do I report a security vulnerability?",
             "Email security@lighthouse.example. A PGP key for encrypting "
             "sensitive reports is published on the security page."),
        ],
    },
]

# (question_id, question_text, [(doc_id, chunk_suffix), ...])
QUESTIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("q01", "How do I invite teammates to my workspace?",
     [("guide-getting-started", "c0")]),
    ("q02", "How do I create a new project board?",
     [("guide-getting-started", "c1")]),
    ("q03", "How do I assign a due date to a task?",
     [("guide-getting-started", "c2")]),
    ("q04", "Can I turn off email notifications?",
     [("guide-getting-started", "c3")]),
    ("q05", "Is there a Lighthouse mobile app for iPhone?",
     [("guide-getting-started", "c4")]),
    ("q06", "What can a Guest user see in Lighthouse?",
     [("guide-permissions", "c4")]),
    ("q07", "What permissions does an Admin have?",
     [("guide-permissions", "c2")]),
    ("q08", "Who can delete the entire workspace?",
     [("guide-permissions", "c1")]),
    ("q09", "How do I connect Lighthouse to Slack?",
     [("guide-integrations", "c0")]),
    ("q10", "Does Lighthouse integrate with GitHub pull requests?",
     [("guide-integrations", "c1")]),
    ("q11", "Is customer data encrypted at rest?",
     [("guide-security", "c0")]),
    ("q12", "Does Lighthouse support single sign-on?",
     [("guide-security", "c1")]),
    ("q13", "What fields are returned when I fetch a single task by ID?",
     [("api-tasks", "c1")]),
    ("q14", "How do I mark a task as done via the API?",
     [("api-tasks", "c3")]),
    ("q15", "What response headers tell me how many API requests I have "
     "left?", [("api-tasks", "c4")]),
    ("q16", "How do I get an access token to call the API?",
     [("api-auth", "c1")]),
    ("q17", "How can I refresh an expired access token without logging in "
     "again?", [("api-auth", "c2")]),
    ("q18", "How do I verify that a webhook request really came from "
     "Lighthouse?", [("api-webhooks", "c2")]),
    ("q19", "What events can trigger a webhook?",
     [("api-webhooks", "c0")]),
    ("q20", "How many times will Lighthouse retry a failing webhook "
     "delivery?", [("api-webhooks", "c3")]),
    ("q21", "How do I install the Python SDK?",
     [("sdk-python", "c0")]),
    ("q22", "How do I create a task using the Python SDK?",
     [("sdk-python", "c2")]),
    ("q23", "How much storage does the Team plan include?",
     [("pricing-table", "c1")]),
    ("q24", "Does the Free plan include single sign-on?",
     [("pricing-table", "c0")]),
    ("q25", "What API rate limit applies to the Enterprise plan?",
     [("pricing-table", "c2")]),
    ("q26", "What operating systems can run the Lighthouse desktop app?",
     [("system-requirements", "c0")]),
    ("q27", "What does a 429 status code mean?",
     [("status-codes", "c1")]),
    ("q28", "What does a 204 response mean when I delete a task?",
     [("status-codes", "c0")]),
    ("q29", "What custom field types can I add to a project?",
     [("field-reference", "c1")]),
    ("q30", "Can I change my Lighthouse plan in the middle of a billing "
     "cycle?", [("faq-billing", "c0")]),
    ("q31", "What payment methods does Lighthouse accept for annual "
     "Enterprise billing?", [("faq-billing", "c1")]),
    ("q32", "Where can I download a copy of my invoice?",
     [("faq-billing", "c2")]),
    ("q33", "If my payment fails and my workspace gets downgraded to Free, "
     "will I lose my project data?",
     [("faq-billing", "c3"), ("faq-billing", "c4")]),
    ("q34", "Can I get my money back after being charged?",
     [("faq-billing", "c5")]),
    ("q35", "How do I reset a forgotten password?",
     [("faq-account", "c0")]),
    ("q36", "Can two separate Lighthouse accounts be merged into one?",
     [("faq-account", "c1")]),
    ("q37", "After I transfer workspace ownership to someone else, what "
     "role do I end up with?", [("faq-account", "c2"), ("faq-account", "c3")]),
    ("q38", "Can an Admin promote a Guest to a full Member?",
     [("faq-account", "c4")]),
    ("q39", "Does the Lighthouse desktop app work without an internet "
     "connection?", [("faq-technical", "c0")]),
    ("q40", "Why do I keep getting rate limited when calling the API?",
     [("faq-technical", "c1")]),
    ("q41", "For how long does Lighthouse keep retrying a webhook delivery "
     "that keeps failing?",
     [("faq-technical", "c2"), ("api-webhooks", "c3")]),
    ("q42", "How do I export all of my Lighthouse data?",
     [("faq-technical", "c3")]),
    ("q43", "Is there a status page I can check during an outage?",
     [("faq-technical", "c4")]),
    ("q44", "Is two-factor authentication required for everyone on my "
     "team?", [("faq-security", "c0"), ("guide-security", "c2")]),
    ("q45", "Where does Lighthouse host customer data by default?",
     [("faq-security", "c1")]),
    ("q46", "How do I take part in Lighthouse's bug bounty and report a "
     "vulnerability I found?",
     [("faq-security", "c2"), ("faq-security", "c3")]),
]


def main() -> None:
    write_corpus(CORPUS_ID, DISPLAY_NAME, DESCRIPTION, DOCUMENTS, QUESTIONS)


if __name__ == "__main__":
    main()
