const host = document.querySelector('#network-scene');

if (host) {
  const run = async () => {
    let THREE;
    try {
      THREE = await import('https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js');
    } catch {
      document.documentElement.classList.add('network-scene-unavailable');
      return;
    }

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false, powerPreference: 'low-power' });
    } catch {
      document.documentElement.classList.add('network-scene-unavailable');
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 0.1, 100);
    camera.position.set(0, 0, 8);
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, innerWidth < 700 ? 1 : 1.35));
    renderer.setSize(innerWidth, innerHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.setAttribute('aria-hidden', 'true');
    host.append(renderer.domElement);
    const codeLanes = innerWidth < 700 ? 8 : 16;
    for (let lane = 0; lane < codeLanes; lane += 1) {
      const code = document.createElement('span');
      code.className = 'code-rain';
      code.textContent = Array.from({ length: 7 }, () => Math.random() > 0.5 ? '1' : '0').join('');
      code.style.setProperty('--code-x', `${(lane + 0.5) * (100 / codeLanes)}%`);
      code.style.setProperty('--code-duration', `${11 + Math.random() * 13}s`);
      code.style.setProperty('--code-delay', `${-Math.random() * 20}s`);
      host.append(code);
    }

    const network = new THREE.Group();
    scene.add(network);
    const globe = new THREE.Group();
    globe.position.set(innerWidth < 700 ? 0.65 : 1.75, 0.05, 0);
    network.add(globe);

    const shell = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.45, 2),
      new THREE.MeshBasicMaterial({ color: 0x00ff41, wireframe: true, transparent: true, opacity: 0.24 })
    );
    globe.add(shell);
    const innerShell = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.1, 1),
      new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true, transparent: true, opacity: 0.16 })
    );
    innerShell.rotation.set(0.4, 0.25, 0.2);
    globe.add(innerShell);

    const nodePositions = [];
    const nodeCount = 38;
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let index = 0; index < nodeCount; index += 1) {
      const y = 1 - (index / (nodeCount - 1)) * 2;
      const radius = Math.sqrt(1 - y * y);
      const angle = golden * index;
      nodePositions.push(new THREE.Vector3(Math.cos(angle) * radius * 1.48, y * 1.48, Math.sin(angle) * radius * 1.48));
    }
    const nodeGeometry = new THREE.BufferGeometry().setFromPoints(nodePositions);
    const nodes = new THREE.Points(nodeGeometry, new THREE.PointsMaterial({ color: 0x00f0ff, size: 0.045, transparent: true, opacity: 0.85, sizeAttenuation: true }));
    globe.add(nodes);

    const edgeVertices = [];
    nodePositions.forEach((point, index) => {
      const nearest = nodePositions
        .map((candidate, candidateIndex) => ({ candidate, candidateIndex, distance: point.distanceToSquared(candidate) }))
        .filter(item => item.candidateIndex > index)
        .sort((a, b) => a.distance - b.distance)
        .slice(0, 2);
      nearest.forEach(({ candidate }) => edgeVertices.push(point.x, point.y, point.z, candidate.x, candidate.y, candidate.z));
    });
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgeVertices, 3));
    const fibers = new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({ color: 0x00ff41, transparent: true, opacity: 0.22 }));
    globe.add(fibers);

    const orbit = new THREE.Mesh(
      new THREE.TorusGeometry(1.9, 0.004, 4, 120),
      new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.24 })
    );
    orbit.rotation.set(1.1, 0.25, -0.35);
    globe.add(orbit);

    const dustCount = innerWidth < 700 ? 180 : 420;
    const dust = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i += 1) {
      dust[i * 3] = (Math.random() - 0.5) * 16;
      dust[i * 3 + 1] = (Math.random() - 0.5) * 10;
      dust[i * 3 + 2] = -1 - Math.random() * 12;
    }
    const dustGeometry = new THREE.BufferGeometry();
    dustGeometry.setAttribute('position', new THREE.BufferAttribute(dust, 3));
    const particleField = new THREE.Points(dustGeometry, new THREE.PointsMaterial({ color: 0x85ffd2, size: 0.015, transparent: true, opacity: 0.4, sizeAttenuation: true }));
    scene.add(particleField);

    const pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };
    let scrollProgress = 0;
    const visibleSections = new Set();
    let lastFrame = 0;
    let destroyed = false;
    let loopRunning = false;
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const setScrollProgress = () => {
      const range = document.documentElement.scrollHeight - innerHeight;
      scrollProgress = range > 0 ? scrollY / range : 0;
    };
    const onPointer = event => {
      pointer.targetX = (event.clientX / innerWidth - 0.5) * 0.5;
      pointer.targetY = (event.clientY / innerHeight - 0.5) * 0.35;
    };
    const onResize = () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(devicePixelRatio || 1, innerWidth < 700 ? 1 : 1.35));
      renderer.setSize(innerWidth, innerHeight);
      globe.position.x = innerWidth < 700 ? 0.65 : 1.75;
    };
    const stopLoop = () => {
      if (!loopRunning) return;
      renderer.setAnimationLoop(null);
      loopRunning = false;
    };
    const startLoop = () => {
      if (!visibleSections.size || document.hidden || reducedMotion.matches) {
        stopLoop();
        renderer.render(scene, camera);
        return;
      }
      if (!loopRunning) {
        loopRunning = true;
        renderer.setAnimationLoop(render);
      }
    };
    const onVisibility = () => { if (document.hidden) stopLoop(); else startLoop(); };
    const render = time => {
      if (destroyed) return;
      const isActive = !document.hidden && visibleSections.size > 0;
      if (!isActive) { stopLoop(); return; }
      if (time - lastFrame >= 33) {
        const dt = Math.min((time - (lastFrame || time)) / 1000, 0.05);
        lastFrame = time;
        pointer.x += (pointer.targetX - pointer.x) * 0.045;
        pointer.y += (pointer.targetY - pointer.y) * 0.045;
        camera.position.x = pointer.x;
        camera.position.y = -pointer.y - scrollProgress * 0.38;
        camera.position.z = 8 - scrollProgress * 0.55;
        camera.lookAt(0, 0, 0);
        if (!reducedMotion.matches) {
          globe.rotation.y += dt * 0.12;
          globe.rotation.x += dt * 0.035;
          innerShell.rotation.y -= dt * 0.08;
          orbit.rotation.z += dt * 0.045;
          particleField.rotation.y += dt * 0.003;
          network.scale.setScalar(1 + scrollProgress * 0.12);
        }
        renderer.render(scene, camera);
      }
    };
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) visibleSections.add(entry.target);
        else visibleSections.delete(entry.target);
      });
      startLoop();
    }, { threshold: 0.01 });
    document.querySelectorAll('main > section, .hero').forEach(section => observer.observe(section));
    addEventListener('pointermove', onPointer, { passive: true });
    addEventListener('scroll', setScrollProgress, { passive: true });
    addEventListener('resize', onResize, { passive: true });
    document.addEventListener('visibilitychange', onVisibility);
    reducedMotion.addEventListener?.('change', startLoop);
    addEventListener('pagehide', () => {
      destroyed = true;
      observer.disconnect();
      renderer.setAnimationLoop(null);
      renderer.dispose();
      scene.traverse(object => {
        object.geometry?.dispose();
        if (Array.isArray(object.material)) object.material.forEach(material => material.dispose());
        else object.material?.dispose();
      });
      removeEventListener('pointermove', onPointer);
      removeEventListener('scroll', setScrollProgress);
      removeEventListener('resize', onResize);
      document.removeEventListener('visibilitychange', onVisibility);
      reducedMotion.removeEventListener?.('change', startLoop);
      host.replaceChildren();
    }, { once: true });
    setScrollProgress();
    renderer.render(scene, camera);
    startLoop();
  };

  run().catch(() => { document.documentElement.classList.add('network-scene-unavailable'); });
}
