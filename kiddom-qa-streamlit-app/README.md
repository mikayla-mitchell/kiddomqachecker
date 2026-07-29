# Kiddom QA Review Studio

A shareable Streamlit app for the Kiddom Issue Annotation Report workflow.
Users can upload one or more report HTML files, apply the established
curriculum-aware classification rules, complete ambiguous decisions in the
browser, and download fully quoted Kiddom import CSVs.

## What the app includes

- Multi-file HTML upload and issue deduplication
- Google Workspace sign-in with an approved-domain allowlist
- Signed-in reviewer attribution kept in a separate shared audit history
- An optional Jira Cloud workspace for finding reviewers and their assigned
  tickets
- Direct, authenticated loading of HTML attachments from Jira tickets
- A confirmed completion handoff that attaches the final CSV, moves the ticket
  to the configured QA status, and reassigns it to the QA owner
- The same math, spelling, capitalization, spacing, punctuation, proper-noun,
  and broken-link rules used by the Codex skill
- An editable human-review queue with source context and Kiddom node links
- Import/export compatibility with `flagged_for_review.csv`
- Final, detailed, review-sheet, and ZIP package downloads
- A Pattern Lab that proposes new rules from saved human decisions
- A shared Report Library that automatically retains compact parsed snapshots
  of every uploaded course
- Shared draft reviews, so another user can reopen a report and continue
- Shared Decision Memory that recognizes the same finding across different
  state/course reports without relying on issue IDs
- Exact and high-confidence near matching with source provenance
- Conflict detection that leaves disputed findings for human review
- Cross-report overlap and course-coverage evidence for safer generalization
- Explicit rule promotion—no silent or automatic learning
- A JSON rulebook that can be version-controlled and synced to the Codex skill

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

Some full-course reports are 650–800 MB. The bundled Streamlit configuration
accepts files up to 1.2 GB and the app clears uploaded binaries after parsing,
but the host still needs enough RAM for the uploaded bytes, decoded HTML, and
parsed findings. For these reports, use a deployment with at least 4 GB RAM;
8 GB is safer when multiple users may process large courses concurrently.

## Share with Streamlit Community Cloud

1. Put this folder in a GitHub repository.
2. In Streamlit Community Cloud, create an app using `app.py` as the entrypoint.
3. Deploy. Google and Jira secrets are optional for local development. Add the
   Google secrets before sharing the production URL.

Streamlit Community Cloud is convenient for smaller reports. Its available
memory can vary, so use a larger container/VM deployment for the largest
full-course HTML files.

Each uploaded HTML is parsed and saved automatically to the shared Report
Library. The app retains the compact structured findings and report metadata,
not the original HTML binary. A full-course snapshot is typically a small
fraction of the 650–800 MB source report.

Clicking **Save visible decisions** also saves a shared draft, so another user
can reopen the report from the library and continue. Completed reviews enter
Decision Memory only when a reviewer clicks **Publish review to shared
memory**.

That database stores compact finding signatures, decisions, correction notes,
and source-report names—not the original HTML. This lets a reviewed course such
as IM v.360 Grade 2 prefill matching findings in New Mexico Grade 2, even when
the report uses different issue IDs, course links, or a state-specific
breadcrumb prefix. A later report can reuse a decision when:

- checker, field, original, and proposed text match; and
- surrounding context is identical, or at least 97% similar.

Conflicting prior decisions are never applied. Reused decisions show their
source and confidence in the review table and remain editable.

By default, shared storage uses `data/shared_memory.sqlite3`. All users of the
same running deployment share it. Set `KIDDOM_SHARED_MEMORY_PATH` to a mounted
persistent volume in production:

```bash
export KIDDOM_SHARED_MEMORY_PATH=/data/kiddom/shared_memory.sqlite3
streamlit run app.py
```

`KIDDOM_DECISION_MEMORY_PATH` remains supported for deployments created with
the earlier decision-only release.

For a team, route everyone to the same Streamlit instance and mount the same
persistent database path. The bundled SQLite/WAL setup is intended for one app
instance serving multiple users. If the deployment later scales to multiple
app replicas, move the storage layer to a shared database service first.

Use the Decision Memory screen to download a portable JSON backup or merge a
backup from another deployment. The backup includes saved report snapshots,
shared drafts, and published decision observations.

To build and persist learning:

1. Upload reports; each parsed snapshot enters the shared Report Library.
2. Save human decisions as shared drafts.
3. Publish a completed review to **Decision Memory** for cross-report reuse.
4. Open **Pattern Lab** to compare human support, occurrences, and course
   coverage.
5. Promote only safe suggestions.
6. Download `kiddom_qa_rules.json`.
7. Replace `rules/base_rules.json` in the repository and redeploy.

Stored but unreviewed reports can increase occurrence and course-coverage
evidence. They never supply a decision label. Only published human decisions
can prefill later reviews or become candidates for rule promotion.

## Google Workspace sign-in

Production access uses Streamlit's OpenID Connect support with Google
Workspace. When `[auth]` is present in deployment secrets, the app becomes a
sign-in gate: reviewers must authenticate before any report, Jira ticket, or
shared memory screen is available. The app verifies the email and applies the
domain allowlist in `[access]`.

Set it up once:

1. In a Google Cloud project owned by the Kiddom Workspace organization, open
   **Google Auth Platform**.
2. Configure the app audience as **Internal**. If you cannot select Internal or
   create credentials, ask a Google Workspace/Cloud administrator to do these
   steps.
3. Create an OAuth client with application type **Web application**.
4. Add exactly one production redirect URI:
   `https://YOUR-APP.streamlit.app/oauth2callback`.
5. Copy the client ID and client secret into the Streamlit app's
   **Settings → Secrets**, along with a long random cookie secret:

```toml
[auth]
redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
cookie_secret = "replace-with-a-long-random-secret"
client_id = "YOUR-GOOGLE-CLIENT-ID.apps.googleusercontent.com"
client_secret = "YOUR-GOOGLE-CLIENT-SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[access]
allowed_email_domains = ["kiddom.co"]
admin_emails = ["app-owner@kiddom.co"]
```

The redirect URI must match in Google Cloud and Streamlit Secrets character for
character. Save the secrets, reboot the app, and open it in a private browser
window to verify the Google sign-in screen and domain restriction.

After sign-in, the Jira workspace presents the shared reviewer list: Karin,
Steve, Janelle, and Mike. Selecting a name loads that person's open tickets.

Local development remains available without `[auth]` and is labeled as local
development mode. Do not share a production deployment until its Google auth
secrets are configured.

## Jira Cloud workflow

Jira is optional. When configured, reviewers can:

1. Open **Jira tickets** in the app.
2. Choose Karin, Steve, Janelle, or Mike from the reviewer list.
3. See open tickets assigned to that person, optionally limited to one project.
4. Open ticket links, use Jira's available status transitions, reassign the
   ticket, or load an attached `.html` / `.htm` report
   directly into the existing QA review pipeline.
5. Complete every human-review decision.
6. In **Exports**, explicitly confirm the Jira handoff.
7. Let the app attach the training-safe final CSV, transition the ticket to the
   configured QA status, and reassign it to the configured QA owner.

The app never fetches arbitrary external ticket links on the server. Jira HTML
attachments are downloaded through Jira's authenticated attachment endpoint;
other links are presented for the reviewer to open.

For an internal team deployment, create a dedicated Jira service account with
only the projects and actions this workflow needs. Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` for local use, or
enter the same secret values in the deployment settings:

```toml
[jira]
base_url = "https://kiddom.atlassian.net"
user_email = "jira-service-account@example.com"
api_token = "..."
project_key = "PMIM"
ready_for_qa_status = "Ready for QA"
qa_account_id = "..."
max_results = 50

[jira_reviewers]
# Optional exact account IDs when Jira finds more than one matching user.
# Karin = "712020:karin-account-id"
# Steve = "712020:steve-account-id"
# Janelle = "712020:janelle-account-id"
# Mike = "712020:mike-account-id"
```

The equivalent environment variables are:

- required: `JIRA_BASE_URL`, `JIRA_USER_EMAIL`, `JIRA_API_TOKEN`
- recommended: `JIRA_PROJECT_KEY`, `JIRA_QA_ACCOUNT_ID`
- optional: `JIRA_READY_FOR_QA_STATUS`, `JIRA_MAX_RESULTS`,
  `JIRA_TICKET_JQL`

`JIRA_TICKET_JQL` can replace the default
`project + assignee + not-done` query. It supports `{account_id}` and
`{project_key}` placeholders. Because it controls ticket visibility, only a
deployment administrator should set it.

The app uses a content hash in the attached CSV filename. If Jira accepts only
part of a handoff, the reviewer can retry without creating a duplicate of the
same CSV; already completed attachment, status, and assignee steps are skipped.
The Jira workflow still depends on the service account having permission to
browse users and issues, add attachments, transition the chosen workflow, and
assign the issue.

Google sign-in identifies and authorizes the reviewer inside this app. Jira
mutations still use the shared service account, so Jira's own change history
will name that integration account. The app separately records the signed-in
reviewer, action, report, and timestamp in shared audit history. That identity
data is never written to the final training CSV.

Keep `.streamlit/secrets.toml` out of source control. It is included in
`.gitignore`. The service account token stays server-side and is never shown in
the app.

## Final-comment policy

The final Kiddom CSV treats `comment` as future QA-tool training data, not as
an audit log. It keeps concrete, reusable explanations and corrections such
as:

- valid curriculum or technical term;
- standards-code, math-notation, or proper-noun false positive;
- confirmed typo or missing-space error; and
- an exact correction such as "Use an ellipsis."

The final export removes AI/checker confidence, missing-reasoning, ambiguity,
workflow, human-review, and reviewer-identity language. Those internal
explanations remain available in the detailed CSV and review interface.
Reviewer identity is stored only in the app's audit history. The same comment
filter is applied before published review notes enter shared Decision Memory.

## Sync approved patterns to the Codex skill

The installed skill can load a rulebook overlay from
`references/app_rules.json`. After downloading a promoted rulebook, run:

```bash
python scripts/sync_rules_to_skill.py /path/to/kiddom_qa_rules.json
```

Use `--skill-dir /custom/path/kiddom-qa-report-review` if the skill is not in
the default `~/.codex/skills` location.

The sync is explicit by design. Reviewers can propose patterns, but an
administrator decides which patterns become production rules.

## Rule behavior

`rules/base_rules.json` contains:

- `protected_spelling_terms`: valid jargon and notation that should be rejected
  as spelling-checker false positives
- `safe_typo_targets`: narrow common-word corrections that are safe to approve
- `exact_rules`: admin-promoted checker/field/original/proposed decisions

Exact rules run before the general classifier. Keep them narrow and promote
only patterns supported by clear evidence.

## Test

```bash
pytest -q
```
