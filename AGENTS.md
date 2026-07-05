# Project Agent Notes

## SSH Access

- Remote access uses SSH public key authentication with passwordless login.
- The SSH public key identity is expected to use this `~/.ssh/config` style entry:

```sshconfig
Host connect.bjb1.seetacloud.com
  HostName connect.bjb1.seetacloud.com
  Port 17591
  User root
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

- Connect to the remote host with:

```sh
ssh -p 17591 root@connect.bjb1.seetacloud.com
```

- Do not expect or request an SSH password for this host. If login fails, check the local SSH key/agent configuration and whether the public key is installed on the remote host.
