 import { getAssetFromKV } from '@cloudflare/kv-asset-handler';

    addEventListener('fetch', event => {
      try {
        event.respondWith(handleEvent(event));
      } catch (e) {
        event.respondWith(new Response('Internal Error', { status: 500 }));
      }
    });

    async function handleEvent(event) {
      try {
        return await getAssetFromKV(event);
      } catch (e) {
        let pathname = new URL(event.request.url).pathname;
        return new Response(`"${pathname}" not found`, {
          status: 404,
          statusText: 'not found',
        });
      }
    }
