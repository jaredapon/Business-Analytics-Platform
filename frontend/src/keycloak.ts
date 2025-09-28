import Keycloak from 'keycloak-js';

const keycloak = new Keycloak({
  url: import.meta.env.VITE_KEYCLOAK_URL,
  realm: import.meta.env.VITE_KEYCLOAK_REALM,
  clientId: import.meta.env.VITE_KEYCLOAK_CLIENT_ID,
});

export const initKeycloak = (onAuthenticated: () => void) => {
  keycloak
    .init({
      onLoad: 'login-required',   
      checkLoginIframe: false,    
      pkceMethod: 'S256',
    })
    .then((ok) => (ok ? onAuthenticated() : keycloak.login()))
    .catch(() => keycloak.login());
};

export const doLogin = keycloak.login;
export const doLogout = keycloak.logout;
export const getToken = () => keycloak.token;
export const isLoggedIn = () => !!keycloak.token;
export const updateToken = (cb: () => void) =>
  keycloak.updateToken(5).then(cb).catch(doLogin);

export default keycloak;
