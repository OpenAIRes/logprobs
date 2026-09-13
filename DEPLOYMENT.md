# Online deployment

This branch is an experimental online-deployment path. `main` remains the local-first version.

## Security model

Do **not** expose `server.py` directly to the internet. The Python service can serve files from the repository root and contains API endpoints that may make paid model calls.

The supplied container instead runs:

```text
Internet -> HTTPS/platform -> Node resampling UI :$PORT
                                  |
                                  +-> Python RecordStore 127.0.0.1:8899
```

Only the Node server is intended to be public. The RecordStore server remains loopback-only.

Before making a deployment public, put authentication in front of the service using your hosting provider, identity-aware proxy, VPN, or reverse proxy. The application itself is not an account system.

## Secrets

Set the OpenAI credential as a platform secret/environment variable:

```text
OPENAI_API_KEY=...
```

Never put it in the image, repository, `local_api_key.json`, or a public environment file.

## Docker

Build:

```bash
docker build -t logprobs-online .
```

Run locally as a deployment test:

```bash
docker run --rm \
  -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  logprobs-online
```

Open `http://localhost:8080`.

The hosting platform should provide HTTPS and route its public port to `$PORT` (8080 by default).

## Persistent data

The current histories are files. Containers commonly have ephemeral filesystems, so production deployment should mount persistent storage for any histories/state that must survive a redeploy. The repository also contains tracked experiment histories; decide explicitly whether the online instance should write into copies on a persistent volume or whether generated records should later be exported/committed.

Do not horizontally scale this container without first moving the writable record store to shared storage or a database. Multiple independent replicas would have divergent histories.

## Public exposure checklist

- Use the `deployment-online` branch, leaving `main` unchanged.
- Configure `OPENAI_API_KEY` as a secret.
- Put authentication in front of the site.
- Use HTTPS.
- Give the service persistent storage if online-generated history must survive redeploys.
- Keep the Python RecordStore port private/loopback-only.
- Set spending/rate limits at the OpenAI project/account level as an additional safeguard.
- Do not expose repository files or `local_api_key.json` through a generic static-file server.

## Hosting

Any Docker-capable host can run this image: a VPS with a reverse proxy, or a managed container platform. The important requirements are one public HTTP port, environment-variable secrets, HTTPS, authentication in front of the app, and persistent disk if the experiment history must survive restarts.

This first deployment branch intentionally does not alter the local behavior on `main`. Once the online path has been tested, deployment-specific improvements can stay here or be merged selectively.
