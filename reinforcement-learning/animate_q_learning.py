"""Render real training excerpts as a classroom GIF; no simulated Q values."""
from pathlib import Path
import os
import tempfile
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from q_learning import MAP, ACTIONS, ACTION_NAMES, START, GOAL, step, update_q, choose_action

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'q-learning-mpl'))

OUT = Path(__file__).parent / 'results'
NAVY, BLUE, GREEN = '#082654', '#1463D6', '#087B3F'
GRAY, PALE, GOLD = '#53657B', '#EAF4FD', '#FFF0B8'
VALID = [s for s in range(25) if MAP[s//5][s%5] != '#']


def font(size, bold=False):
    from matplotlib import font_manager
    prop = font_manager.FontProperties(family='DejaVu Sans', weight='bold' if bold else 'normal')
    return ImageFont.truetype(font_manager.findfont(prop), size)

FONTS = {(s,b):font(s,b) for s in (16,18,20,22,24,28,36,42) for b in (False,True)}


def collect():
    rng = np.random.default_rng(7)
    q = np.zeros((25,4))
    episodes, first_success = {}, None
    for ep in range(1,1501):
        epsilon = max(.05, .995**(ep-1))
        state, path, records = START, [START], []
        for t in range(1,101):
            # Exactly the same RNG calls and tie-breaking as choose_action.
            explore = rng.random() < epsilon
            action = int(rng.integers(4)) if explore else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            ns, reward, done = step(state,action)
            before = q.copy()
            target = reward if done else reward+.95*q[ns].max()
            update_q(q,state,action,reward,ns,done,.2,.95)
            records.append(dict(ep=ep,t=t,epsilon=epsilon,s=state,a=action,ns=ns,r=reward,
                                done=done,mode='Explore' if explore else 'Exploit',before=before,
                                after=q.copy(),target=float(target),path=path.copy(),evaluation=False))
            path.append(ns)
            state=ns
            if done:
                if first_success is None: first_success=ep
                break
        if ep in (1,50,1500) or ep==first_success:
            episodes[ep]=records
    saved=OUT/'q_table.npy'
    if saved.exists():
        np.testing.assert_array_equal(q,np.load(saved))
    groups=[('Early exploration | episode 1, first 8 steps',episodes[1][:8]),
            (f'First goal reached | episode {first_success}, final 8 steps',episodes[first_success][-8:]),
            ('Learning continues | episode 50, final 8 steps',episodes[50][-8:]),
            ('Late training | episode 1500, complete episode',episodes[1500])]
    erng=np.random.default_rng(123)
    state,path,records=START,[START],[]
    for t in range(1,101):
        a=choose_action(q,state,0,erng)
        ns,r,done=step(state,a)
        records.append(dict(ep=1500,t=t,epsilon=0,s=state,a=a,ns=ns,r=r,done=done,
                            mode='Greedy evaluation',before=q.copy(),after=q.copy(),target=0,
                            path=path.copy(),evaluation=True))
        path.append(ns); state=ns
        if done: break
    assert done and len(records)==8
    groups.append(('Final policy | 8 steps to the goal, no learning updates',records))
    # Verify all displayed records update only the selected state-action pair.
    for _, records in groups:
        for r in records:
            expected=r['before'].copy()
            if not r['evaluation']:
                update_q(expected,r['s'],r['a'],r['r'],r['ns'],r['done'],.2,.95)
            np.testing.assert_array_equal(expected,r['after'])
    return groups


def render(title,r,phase,show_title=True,compact=False):
    im=Image.new('RGB',(1600,900),'white'); d=ImageDraw.Draw(im)
    def text(x,y,value,size=22,color=NAVY,bold=False):
        d.text((x,y),value,font=FONTS[size,bold],fill=color)
    if show_title and not compact:
        text(55,25,'Q-Learning: Explore, Act, Update',42,bold=True)
    if not compact:
        text(55,85,title,24,GRAY)
    text(55,127,f"Step {r['t']}   |   epsilon = {r['epsilon']:.3f}   |   {r['mode']}",22,BLUE,True)
    text(55,166,'1. Choose an action' if phase=='before' else ('2. Observe the result' if r['evaluation'] else '2. Observe and update one Q-value'),24,bold=True)
    # Exact maze; robot icon built from geometric shapes.
    x0,y0,cell=75,225,88
    path=r['path']+([r['ns']] if phase=='after' else [])
    for s in range(25):
        row,col=divmod(s,5); x,y=x0+col*cell,y0+row*cell
        fill='#283444' if MAP[row][col]=='#' else ('#BCEED2' if s==GOAL else PALE)
        d.rectangle((x,y,x+cell,y+cell),fill=fill,outline='white',width=3)
        if MAP[row][col]!='#': text(x+6,y+4,f'({row},{col})',16,GRAY)
    points=[(x0+(s%5+.5)*cell,y0+(s//5+.5)*cell) for s in path]
    if len(points)>1: d.line(points,fill='#93B8E9',width=5)
    text(x0+4*cell+17,y0+4*cell+61,'GOAL',16,GREEN,True)
    robot=r['s'] if phase=='before' else r['ns']
    cx,cy=x0+(robot%5+.5)*cell,y0+(robot//5+.5)*cell
    d.rounded_rectangle((cx-23,cy-17,cx+23,cy+19),8,fill=BLUE,outline=NAVY,width=2)
    d.line((cx,cy-17,cx,cy-26),fill=NAVY,width=3)
    d.ellipse((cx-4,cy-32,cx+4,cy-24),fill=BLUE)
    for dx in (-9,9): d.ellipse((cx+dx-3,cy-6,cx+dx+3,cy),fill='white')
    d.arc((cx-9,cy-2,cx+9,cy+12),0,180,fill='white',width=2)
    if phase=='before':
        dr,dc=ACTIONS[r['a']]; ex,ey=cx+dc*38,cy+dr*38
        d.line((cx,cy,ex,ey),fill='#E89616',width=5)
        d.polygon([(ex,ey),(ex-dc*10-dr*6,ey-dr*10+dc*6),(ex-dc*10+dr*6,ey-dr*10-dc*6)],fill='#E89616')
    text(75,682,f"Action: {ACTION_NAMES[r['a']].title()}   |   Reward: {r['r']:+d}" if phase=='after' else f"Selected action: {ACTION_NAMES[r['a']].title()}",24,BLUE,True)
    if phase=='after':
        status='Goal reached: episode ends.' if r['done'] else ('Blocked: the robot stays in place.' if r['s']==r['ns'] else f"Next state: {divmod(r['ns'],5)}")
        text(75,719,status,20,GREEN if r['done'] else GRAY)
    # Full non-obstacle table, with all four actions in code order.
    tx,ty=650,208; widths=[170,155,155,155,155]; rh=24
    text(tx,166,'Q-table',24,bold=True)
    text(tx+160,170,'Obstacle rows omitted; goal values remain 0.',16,GRAY)
    for i,label in enumerate(['State','Up','Right','Down','Left']):
        x=tx+sum(widths[:i]); d.rectangle((x,ty,x+widths[i],ty+30),fill=NAVY)
        text(x+12,ty+3,label,18,'white',True)
    q=r['before'] if phase=='before' else r['after']
    for i,s in enumerate(VALID):
        y=ty+30+i*rh
        fill=PALE if s==r['s'] else ('#F5F7FA' if i%2==0 else 'white')
        d.rectangle((tx,y,tx+sum(widths),y+rh),fill=fill)
        text(tx+12,y+1,str(divmod(s,5))+('  goal' if s==GOAL else ''),16,GRAY)
        for a in range(4):
            x=tx+widths[0]+a*155
            selected=s==r['s'] and a==r['a']
            if selected: d.rectangle((x,y,x+155,y+rh),fill=GOLD if phase=='before' else '#BCEED2')
            text(x+20,y+1,f'{q[s,a]:.3f}',16,NAVY,selected)
    text(tx,ty+30+len(VALID)*rh+10,'Highlighted cell: selected state-action pair',16,GRAY)
    d.rounded_rectangle((55,773,1545,875),16,fill=PALE)
    if r['evaluation']:
        text(80,790,'Evaluation: choose a highest-Q action; keep the Q-table fixed.',24,bold=True)
        text(80,834,'Exploration is switched off (epsilon = 0).',20,GRAY)
    else:
        old=r['before'][r['s'],r['a']]; new=r['after'][r['s'],r['a']]
        target='10.000 (terminal state: no future value)' if r['done'] else f"{r['r']:.3f} + 0.95 x {r['before'][r['ns']].max():.3f} = {r['target']:.3f}"
        text(80,787,'Target = '+target,22)
        text(80,830,f"Q{divmod(r['s'],5)}, {ACTION_NAMES[r['a']]}:  {old:.3f} + 0.20 x ({r['target']:.3f} - ({old:.3f})) = {new:.3f}" if phase=='after' else 'Take the action, observe the reward and next state, then update.',22,BLUE,True)
    return im.crop((0,110,1600,890)) if compact else im


def export_parts(groups):
    """Five independent, compact clips without titles or transition cards."""
    import subprocess
    import imageio_ffmpeg
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    folder = OUT / 'video_parts'
    folder.mkdir(exist_ok=True)
    names = ['01_early_exploration', '02_first_goal', '03_learning_continues',
             '04_late_training', '05_final_policy']
    details = []
    for name, (title, records) in zip(names, groups):
        destination = folder / f'{name}.mp4'
        command = [encoder, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1600x780', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264',
                   '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(destination)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        duration = 0
        try:
            for r in records:
                for phase in ('before', 'after'):
                    frame = render(title, r, phase, show_title=False, compact=True)
                    milliseconds = 650 if phase == 'before' else (2100 if r['done'] else 1150)
                    duration += milliseconds
                    pixels = frame.tobytes()
                    for _ in range(round(milliseconds / 50)):
                        process.stdin.write(pixels)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f'Encoding failed: {destination}')
        subprocess.run([encoder, '-v', 'error', '-i', str(destination), '-f', 'null', '-'], check=True)
        render(title, records[-1], 'after', show_title=False, compact=True).save(folder / f'{name}.png')
        details.append({'file': destination.name, 'section': title, 'duration_seconds': duration/1000,
                        'width': 1600, 'height': 780, 'fps': 20})
        print(f'Validated {destination.name}: {duration/1000:.2f} s', flush=True)
    (folder / 'manifest.json').write_text(json.dumps(details, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='store_true', help='Export MP4 without the main title (requires imageio-ffmpeg).')
    parser.add_argument('--split-videos', action='store_true', help='Export five compact MP4 clips without titles or transition cards.')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    groups=collect(); frames=[]; durations=[]; manifest=[]
    if args.split_videos:
        export_parts(groups)
        return
    for title,records in groups:
        # Explicit transition card: excerpts are not presented as consecutive steps.
        card=Image.new('RGB',(1600,900),'white'); d=ImageDraw.Draw(card)
        d.text((90,260),'TRAINING REPLAY' if not records[0]['evaluation'] else 'EVALUATION',font=FONTS[24,True],fill=BLUE)
        d.text((90,335),title,font=FONTS[28,True],fill=NAVY)
        d.text((90,410),'Selected excerpts from one real run (seed = 7).',font=FONTS[24,False],fill=GRAY)
        d.text((90,460),'Training continues during omitted steps and episodes.',font=FONTS[24,False],fill=GRAY)
        frames.append(card);durations.append(2300)
        for r in records:
            for phase in ('before','after'):
                frames.append(render(title,r,phase,show_title=not args.video)); durations.append(650 if phase=='before' else (2100 if r['done'] else 1150))
            manifest.append({k:v for k,v in r.items() if k not in ('before','after','path')})
    if args.video:
        import subprocess
        import imageio_ffmpeg
        encoder = imageio_ffmpeg.get_ffmpeg_exe()
        destination = OUT/'q_learning_animation_no_title.mp4'
        command = [encoder, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1600x900', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264',
                   '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(destination)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame, duration in zip(frames, durations):
                pixels = frame.tobytes()
                for _ in range(round(duration / 50)):
                    process.stdin.write(pixels)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoding failed')
        subprocess.run([encoder, '-v', 'error', '-i', str(destination), '-f', 'null', '-'], check=True)
        frames[2].save(OUT/'q_learning_video_preview.png')
        print(f'Validated MP4: 1600x900, 20 fps, {sum(durations)/1000:.1f} seconds. {destination}')
        return
    frames[0].save(OUT/'q_learning_animation.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
    # Slow replay of the first goal approach, with substantial early Q updates.
    detail=groups[1]; slow=[]; times=[]
    for r in detail[1]:
        for phase in ('before','after'):
            slow.append(render(detail[0],r,phase));times.append(1400 if phase=='before' else 2600)
    slow[0].save(OUT/'q_learning_step_by_step.gif',save_all=True,append_images=slow[1:],duration=times,loop=0,optimize=False)
    render(groups[1][0],groups[1][1][-1],'after').save(OUT/'q_learning_animation_preview.png')
    (OUT/'animation_trace.json').write_text(json.dumps(manifest,indent=2))
    with Image.open(OUT/'q_learning_animation.gif') as gif:
        assert gif.n_frames==len(frames)
        print(f'{gif.n_frames} frames; {sum(durations)/1000:.1f} seconds; 1600x900')
    print('Q-table exactly matches the original 1500-episode run; every displayed update verified.')

if __name__=='__main__': main()
