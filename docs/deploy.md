# Putting the demo on targeting.peterparker.ca

The demo is `demo/`: five static files, a `fonts/` directory and a `data/` directory. There is
no build step, no backend and no dependency to install, so hosting it is a file copy and a DNS
record.

PLAN.md section 7 specifies Azure Static Web Apps on the free tier, because this project is
the portfolio's Azure hosting example. GitHub Pages is the fallback if the free tier changes,
and is at the bottom of this page.

## Before you start

- The repository is private. Both routes below work with a private repository, but they
  differ in what they put in it: route A commits a GitHub Actions workflow, route B does not.
- The free tier includes custom domains and enough bandwidth for this page, which is about
  400 KB of JSON, 180 KB of fonts and 25 KB of HTML, CSS and JavaScript.
- `demo/staticwebapp.config.json` is already committed. It sets a content security policy
  that allows the page's own script, stylesheet and fonts and nothing else, so the deployment
  cannot quietly start loading anything off-origin. The two typefaces are peterparker.ca's,
  copied into `demo/fonts/` rather than linked to that host, which is why `font-src 'self'`
  is in the policy and no other font source is.
- **Check the page with `uv run itx demo serve`, not with `python -m http.server`.** The
  second sends no headers, so it shows you a page the policy would partly refuse. That is not
  hypothetical: the legend's colour swatches were blank on the live site for two weeks and
  correct in every local check, because a style attribute in markup is what `style-src 'self'`
  blocks. `itx demo serve` sends the config's own headers and the defect is visible at once.

## Route A: deploy from GitHub, which redeploys on every push

Use this if you want the live page to track `main` without anyone remembering to run
anything.

1. In the Azure Portal, create a resource of type **Static Web App**. Give it a resource
   group, a name, the **Free** plan, and a region near you. Region affects only where the
   build runs; the content is served from a CDN either way.
2. For deployment, choose **GitHub**, authorise Azure, and select
   `Peter-A-P/intervention-targeting-engine` and the `main` branch.
3. Build details: choose the **Custom** preset and set

   - **App location:** `demo`
   - **Api location:** leave empty
   - **Output location:** leave empty

   The output location matters. The demo is already built, so pointing it at a build output
   directory that does not exist is the usual way this fails.
4. Create it. Azure commits a workflow to `.github/workflows/` and runs it. The first
   deployment takes a few minutes; watch it under Actions on GitHub.
5. Azure gives you a URL like `https://<random-name>.azurestaticapps.net`. Open it and check
   the dataset picker lists five datasets and the slider moves.

## Route B: deploy from this machine, which puts nothing in the repository

Use this if you would rather not have a workflow file and an Azure credential in a repository
that is about to go public.

1. Create the Static Web App as above, but choose **Other** instead of GitHub for
   deployment. No repository is connected.
2. In the resource, open **Overview** and copy the **deployment token**. Treat it as a
   secret: it is enough on its own to publish to that site. Do not put it in the repository,
   in `CLAUDE.local.md`, or in a shell history you keep.
3. Put the token in the environment for this shell only, straight from the CLI, so it
   never appears on a command line or in shell history:

   ```powershell
   $env:SWA_CLI_DEPLOYMENT_TOKEN = az staticwebapp secrets list --name <app> --resource-group <group> --subscription <sub> --query properties.apiKey -o tsv
   ```

4. Deploy from the repository root through `npx`, which needs Node and nothing else:

   ```powershell
   npx --yes @azure/static-web-apps-cli deploy ./demo --env production
   ```

   A global `npm install -g` also works, but on this machine npm's global folder is not on
   PATH, so the installed `swa` command is not found until that is fixed; `npx` sidesteps it.

5. Redeploy with the same two commands whenever `itx demo build` has rewritten `demo/data/`.

## The custom domain, either route

1. In the Static Web App, open **Custom domains** and **Add**. Enter
   `targeting.peterparker.ca`.
2. Azure asks for a **CNAME** record and shows the target, which is the
   `<name>.azurestaticapps.net` hostname. A subdomain takes a CNAME; only an apex domain
   needs the TXT validation dance, and this is a subdomain, so it is one record.
3. At whichever DNS provider holds `peterparker.ca`, add:

   | Type | Name | Value | TTL |
   |---|---|---|---|
   | CNAME | `targeting` | `<name>.azurestaticapps.net` | 3600 |

4. Back in Azure, click validate. Propagation is usually minutes and occasionally an hour.
   Azure issues and renews the TLS certificate itself once validation passes; there is
   nothing to buy or install.
5. Check `https://targeting.peterparker.ca` serves the page over HTTPS.

## As deployed, 2026-09-14

Route B. A second Static Web App, `targeting-peterparker-ca`, free tier, in the same
resource group as the portfolio site's app, created with `az staticwebapp create` and
published with the `npx` command above. Custom domain `targeting.peterparker.ca` attached
with `az staticwebapp hostname set`, which validated by the CNAME alone and reported
`Ready` within the command's own wait. The CNAME is at Cloudflare with proxy status
"DNS only"; a proxied record would have blocked validation and the managed certificate.
The live page returns the content security policy from `staticwebapp.config.json` and
`data/index.json` lists five datasets.

Why a second app rather than a second hostname on the existing one: a Static Web App serves
one set of files to every hostname attached to it and does not route by host, so a subdomain
with different content is a separate app. The alternative was a path under the portfolio
site, which would have changed the address the plan names, dropped this page's own security
policy in favour of the site's, and coupled the two repositories' builds.

The README, the site's stand-in page and PLAN.md section 10 carry the URL.

## If the free tier changes: GitHub Pages

Pages serves a subdirectory only from a branch root, so `demo/` has to become the root of a
published branch:

```bash
git subtree push --prefix demo origin gh-pages
```

Then enable Pages on the `gh-pages` branch in the repository settings and add the same CNAME
record, pointing at `peter-a-p.github.io` instead. Two things are worse this way: Pages
requires the repository to be public before it will serve, which forces the week 8 order, and
`staticwebapp.config.json` is ignored, so the content security policy above is lost.
