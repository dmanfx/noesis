const char* px_proxy_factory_get_proxies(void* factory, const char* url) {
    (void)factory;
    (void)url;
    return "direct://";
}
