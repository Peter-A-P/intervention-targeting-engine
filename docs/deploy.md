# Putting the demo on targeting.peterparker.ca

The demo is `demo/`: four static files and a `data/` directory. There is no build step, no
backend and no dependency to install, so hosting it is a file copy and a DNS record.

PLAN.md section 7 specifies Azure Static Web Apps on the free tier, because this project is
the portfolio's Azure hosting example. GitHub Pages is the fallback if the free tier changes,
and is at the bottom of this page.

## Before you start

- The repository is private. Both routes below work with a private repository, but they
  differ in what they put in it: route A commits a GitHub Actions workflow, route B does not.
- The free tier includes custom domains and enough bandwidth for this page, which is about
  400 KB of JSON plus 20 KB of HTML, CSS and JavaScript.
- `demo/staticwebapp.config.json` is already committed. It sets a content security policy
  that allows the page's own script and stylesheet and nothing else, so the deployment
  cannot quietly start loading anything off-origin.

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
3. Install the CLI once, with Node available:

   ```bash
   npm install -g @azure/static-web-apps-cli
   ```

4. Deploy from the repository root:

   ```bash
   swa deploy ./demo --deployment-token <the token> --env production
   ```

5. Redeploy with the same command whenever `itx demo build` has rewritten `demo/data/`.

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

## Afterwards

- Put the URL in the README, next to the results tables, and in
  `../peterparker.ca/content/01-intervention-targeting.md`, which currently says the demo is
  a directory you can open rather than a link you can click.
- Tick the demo line in PLAN.md section 10, which is currently marked partial because "live"
  is what that line asks for and a built page is not a live one.

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
