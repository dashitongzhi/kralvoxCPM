# Project Agent Notes

## SSH Access

- Remote access uses SSH public key authentication with passwordless login.
- Connect to the remote host with:

```sh
ssh -p 17591 root@connect.bjb1.seetacloud.com
```

- Do not expect or request an SSH password for this host. If login fails, check the local SSH key/agent configuration and whether the public key is installed on the remote host.
